#!/usr/bin/env python3
"""Genera paneles visuales con anotaciones para revisión semántica, sin modificar originales."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps

from auditar_datasets import IMAGE_EXTS, detect_mask, is_yolo_label, pick_images, split_name, relpath


def safe_text(s: str, max_chars=45) -> str:
    s = str(s).replace("\n", " ")
    # La fuente de sistema de Pillow no siempre admite caracteres Unicode; mantener ASCII en overlay.
    s = s.encode("ascii", errors="replace").decode("ascii")
    return s[:max_chars] + ("..." if len(s) > max_chars else "")


def collect_examples(root: Path) -> list[dict]:
    files = [p for p in root.rglob("*") if p.is_file()]
    images = [p for p in files if p.suffix.lower() in IMAGE_EXTS and not detect_mask(p)]
    by_name = defaultdict(list)
    by_stem = defaultdict(list)
    for p in images:
        by_name[p.name.lower()].append(p)
        by_stem[p.stem.lower()].append(p)
    sample = defaultdict(lambda: {"rects": [], "poly": [], "classes": set(), "source": set()})
    for p in files:
        if p.suffix.lower() != ".json":
            continue
        try:
            info = json.loads(p.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        if not (isinstance(info, dict) and isinstance(info.get("images"), list)
                and isinstance(info.get("annotations"), list)):
            continue
        refs = {}
        category = {str(c.get("id")): str(c.get("name", c.get("id")))
                    for c in info.get("categories", []) if isinstance(c, dict)}
        for im in info["images"]:
            hits = pick_images(str(im.get("file_name", "")), by_name, p, root)
            if len(hits) == 1:
                refs[str(im.get("id"))] = hits[0]
        for a in info["annotations"]:
            path = refs.get(str(a.get("image_id")))
            if path is None:
                continue
            obj = sample[path]
            cls = category.get(str(a.get("category_id")), str(a.get("category_id")))
            obj["classes"].add(cls)
            obj["source"].add("COCO")
            bb = a.get("bbox")
            if isinstance(bb, (list, tuple)) and len(bb) == 4:
                try:
                    x, y, w, h = map(float, bb)
                    obj["rects"].append((x, y, x+w, y+h, cls, "abs"))
                except (ValueError, TypeError):
                    pass
            sg = a.get("segmentation")
            if isinstance(sg, list):
                for segment in sg:
                    if isinstance(segment, list) and len(segment) >= 6 and len(segment) % 2 == 0:
                        try:
                            coords = [float(v) for v in segment]
                            obj["poly"].append((coords, "abs"))
                        except (TypeError, ValueError):
                            pass
    for p in files:
        if p.suffix.lower() != ".xml":
            continue
        try:
            a = ET.parse(p).getroot()
        except (ET.ParseError, OSError):
            continue
        if a.tag.lower() != "annotation":
            continue
        f = a.findtext("filename") or p.stem
        if Path(f).suffix:
            hits = pick_images(f, by_name, p, root)
        else:
            hits = by_stem[f.lower()]
        if len(hits) != 1:
            continue
        path = hits[0]
        obj = sample[path]
        for part in a.findall("object"):
            name = part.findtext("name") or "SIN_NOMBRE"
            obj["classes"].add(name)
            obj["source"].add("VOC")
            bb = part.find("bndbox")
            if bb is not None:
                try:
                    coords = [float(bb.findtext(n)) for n in ("xmin", "ymin", "xmax", "ymax")]
                    obj["rects"].append((*coords, name, "abs"))
                except (ValueError, TypeError):
                    pass
    yolo_names = {}
    for p in files:
        if p.suffix.lower() not in (".yaml", ".yml"):
            continue
        try:
            parsed = yaml.safe_load(p.read_text(encoding="utf-8-sig"))
            name_data = parsed.get("names", {}) if isinstance(parsed, dict) else {}
            if isinstance(name_data, list):
                yolo_names = {str(i): str(n) for i, n in enumerate(name_data)}
            elif isinstance(name_data, dict):
                yolo_names = {str(i): str(n) for i, n in name_data.items()}
            if yolo_names:
                break
        except (OSError, yaml.YAMLError):
            pass
    for p in files:
        if not is_yolo_label(p):
            continue
        hits = by_stem.get(p.stem.lower(), [])
        if len(hits) > 1:
            split = split_name(relpath(p, root))
            subset = [h for h in hits if split_name(relpath(h, root)) == split]
            if subset:
                hits = subset
        if len(hits) != 1:
            continue
        obj = sample[hits[0]]
        try:
            rows = p.read_text(encoding="utf-8-sig").splitlines()
        except (UnicodeError, OSError):
            continue
        for row in rows:
            cols = row.split()
            try:
                k = str(int(cols[0])); pts = [float(x) for x in cols[1:]]
            except (ValueError, IndexError):
                continue
            cls = yolo_names.get(k, f"ID_{k}")
            obj["classes"].add(cls)
            obj["source"].add("YOLO")
            if len(pts) == 4:
                x, y, w, h = pts
                obj["rects"].append((x-w/2, y-h/2, x+w/2, y+h/2, cls, "norm"))
            elif len(pts) >= 6 and len(pts) % 2 == 0:
                obj["poly"].append((pts, "norm"))
    for img in images:
        if img not in sample:
            sample[img]  # muestra de imágenes sin anotaciones cuando haga falta
    return [{"path": p, **item} for p, item in sorted(sample.items(), key=lambda item: str(item[0]).lower())]


def choose_stratified(examples: list[dict], limit: int) -> list[dict]:
    if limit <= 0:
        return []
    chosen: list[dict] = []
    seen_classes: set[str] = set()
    used: set[Path] = set()
    # Priorizar al menos un ejemplo por cada clase antes de rellenar.
    for obj in examples:
        if len(chosen) >= limit:
            break
        new_classes = obj["classes"] - seen_classes
        if new_classes:
            chosen.append(obj)
            used.add(obj["path"])
            seen_classes.update(obj["classes"])
    remaining = [x for x in examples if x["path"] not in used]
    if remaining and len(chosen) < limit:
        slots = min(limit - len(chosen), len(remaining))
        indices = [round(i * (len(remaining) - 1) / max(1, slots - 1)) for i in range(slots)]
        chosen.extend(remaining[i] for i in indices)
    return chosen


def build_contact_sheet(examples: list[dict], output: Path, title: str):
    if not examples:
        return None
    tw, th, caption_h, pad = 360, 275, 47, 12
    cols = min(4, len(examples))
    rows = math.ceil(len(examples) / cols)
    canvas = Image.new("RGB", (cols * tw, rows * th + 36), "white")
    dr = ImageDraw.Draw(canvas)
    dr.text((12, 9), safe_text(title, 120), fill="black")
    for i, item in enumerate(examples):
        p = item["path"]
        x0 = (i % cols) * tw
        y0 = (i // cols) * th + 36
        try:
            with Image.open(p) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
            iw, ih = image.size
            maxw, maxh = tw - 2 * pad, th - caption_h - 2 * pad
            ratio = min(maxw / iw, maxh / ih)
            nw, nh = round(iw * ratio), round(ih * ratio)
            scaled = image.resize((nw, nh), Image.Resampling.LANCZOS)
            canvas.paste(scaled, (x0 + pad, y0 + pad))
            overlay = ImageDraw.Draw(canvas)
            for l, t, r, b, label, mode in item["rects"][:50]:
                if mode == "norm":
                    l, r = l * iw, r * iw
                    t, b = t * ih, b * ih
                box = (x0 + pad + l * ratio, y0 + pad + t * ratio,
                       x0 + pad + r * ratio, y0 + pad + b * ratio)
                overlay.rectangle(box, outline="#ed3939", width=2)
                overlay.text((box[0], max(y0 + pad, box[1] - 13)), safe_text(label, 20), fill="#e52020")
            for points, mode in item["poly"][:50]:
                if len(points) < 6:
                    continue
                coords = []
                for j in range(0, len(points), 2):
                    xx, yy = points[j], points[j+1]
                    if mode == "norm":
                        xx *= iw
                        yy *= ih
                    coords.append((x0 + pad + xx * ratio, y0 + pad + yy * ratio))
                overlay.line(coords + [coords[0]], fill="#e7a500", width=3)
        except (OSError, ValueError, SyntaxError):
            overlay = ImageDraw.Draw(canvas)
            overlay.text((x0 + pad, y0 + pad + 12), "NO SE PUDO ABRIR LA IMAGEN", fill="red")
        dr = ImageDraw.Draw(canvas)
        dr.text((x0 + pad, y0 + th - caption_h), safe_text(p.name, 43), fill="black")
        desc = ", ".join(sorted(item["classes"])) or "sin anotaciones vinculadas"
        dr.text((x0 + pad, y0 + th - caption_h + 17), safe_text(desc, 43), fill="#444444")
        dr.rectangle((x0, y0, x0 + tw - 1, y0 + th - 1), outline="#bbbbbb")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=88)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description="Muestrarios visuales no destructivos por dataset")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--limite-por-dataset", type=int, default=16)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    config = json.loads((root / "config" / "datasets.json").read_text(encoding="utf-8"))
    output = (root / config.get("directorio_resultados", "resultados/auditoria") / "muestrarios").resolve()
    if not output.is_relative_to(root):
        parser.error("Directorio de resultados fuera de la raíz")
    for d in config["datasets"]:
        folder = (root / d["ruta"]).resolve()
        if not folder.is_relative_to(root) or folder in output.parents or output in folder.parents:
            parser.error(f"Rutas no seguras para {d['nombre']}")
        if not folder.is_dir():
            print(f"[OMITIDO] {d['nombre']}: carpeta ausente")
            continue
        print(f"[VISUAL] Buscando muestras anotadas: {d['nombre']}", flush=True)
        examples = choose_stratified(collect_examples(folder), args.limite_por_dataset)
        safe_name = "".join(c if c.isascii() and c.isalnum() else "_" for c in d["nombre"])
        out = build_contact_sheet(examples, output / f"{safe_name}_muestrario.jpg", d["nombre"])
        print(f"[LISTO] {out or 'Sin imágenes'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
