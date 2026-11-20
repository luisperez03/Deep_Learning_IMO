#!/usr/bin/env python3
"""Auditoría exploratoria no destructiva de datasets COCO/YOLO/VOC para tesis IMO.

Uso: python scripts/auditar_datasets.py [--root RUTA] [--todas-imagenes] [--hash-duplicados]
Solo escribe dentro de `resultados/auditoria` y nunca modifica los datos de entrada.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

try:
    from PIL import Image
except ImportError:
    sys.exit("Falta Pillow. Instala las dependencias: python -m pip install -r requirements.txt")

try:
    import yaml
except ImportError:
    sys.exit("Falta PyYAML. Instala las dependencias: python -m pip install -r requirements.txt")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif"}
MASK_DIRS = {"segmentationclass", "segmentationobject", "masks", "mask", "segmentationmasks", "semantic_masks", "instance_masks"}
VALID_SPLITS = {"train": "train", "training": "train", "val": "val", "valid": "val", "validation": "val", "test": "test", "testing": "test"}


def split_name(path: Path) -> str:
    """Detecta particiones solo por componentes completos del directorio."""
    for x in path.parts:
        if x.lower() in VALID_SPLITS:
            return VALID_SPLITS[x.lower()]
    # Un JSON COCO puede llamarse train.json sin tener carpeta train
    low = path.stem.lower()
    if re.search(r"(^|[_\-.])(train|training)([_\-.]|$)", low):
        return "train"
    if re.search(r"(^|[_\-.])(valid|validation|val)([_\-.]|$)", low):
        return "val"
    if re.search(r"(^|[_\-.])(test|testing)([_\-.]|$)", low):
        return "test"
    return "sin_identificar"


def rel(p: Path, root: Path) -> str:
    try:
        return p.relative_to(root).as_posix()
    except ValueError:
        return str(p)


def detect_mask(p: Path) -> bool:
    return any(c.lower() in MASK_DIRS for c in p.parent.parts)


def is_yolo_label(p: Path) -> bool:
    if p.suffix.lower() != ".txt":
        return False
    return any(part.lower() == "labels" for part in p.parent.parts)


def pick_images(ref: str, index_name: dict[str, list[Path]], source: Path, dataset_root: Path) -> list[Path]:
    """Devuelve 0/1/N candidatos; nunca escoge arbitrariamente entre ambiguos."""
    norm = ref.replace("\\", "/")
    basename = Path(norm).name.lower()
    candidates = index_name.get(basename, [])
    if len(candidates) <= 1:
        return candidates
    # Desambiguar por ruta interna y por split de anotación.
    normalized_ref = norm.lstrip("./").lower()
    suffix_matches = [p for p in candidates if rel(p, dataset_root).lower().endswith(normalized_ref)]
    if len(suffix_matches) == 1:
        return suffix_matches
    if suffix_matches:
        candidates = suffix_matches
    s = split_name(relpath(source, dataset_root))
    if s != "sin_identificar":
        split_matches = [p for p in candidates if split_name(relpath(p, dataset_root)) == s]
        if len(split_matches) == 1:
            return split_matches
        if split_matches:
            candidates = split_matches
    # Preferir cercanía del archivo de anotación si no se puede resolver aún.
    distances = [(len(set(p.parent.parts) ^ set(source.parent.parts)), p) for p in candidates]
    best_score = min(d for d, _ in distances)
    best = [p for d, p in distances if d == best_score]
    return best


def relpath(path: Path, root: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return path


def load_yolo_names(folder: Path, files: list[Path], warnings: list[dict], dataset: str) -> dict[int, str]:
    candidates = [f for f in files if f.suffix.lower() in (".yaml", ".yml")]
    names_options: list[tuple[Path, dict[int, str]]] = []
    for f in candidates:
        try:
            obj = yaml.safe_load(f.read_text(encoding="utf-8-sig"))
            if not isinstance(obj, dict) or "names" not in obj:
                continue
            value = obj["names"]
            names: dict[int, str] = {}
            if isinstance(value, list):
                names = {i: str(name) for i, name in enumerate(value)}
            elif isinstance(value, dict):
                names = {int(key): str(name) for key, name in value.items()}
            if names:
                names_options.append((f, names))
        except (OSError, ValueError, yaml.YAMLError, TypeError) as exc:
            warnings.append(issue(dataset, "AVISO", f, f"No fue posible leer YAML: {exc}", folder))
    if len(names_options) > 1:
        unique = {tuple(sorted(m.items())) for _, m in names_options}
        if len(unique) > 1:
            warnings.append(issue(dataset, "REVISAR", folder, "Varios archivos YAML contienen mapas de clase distintos; revisar manualmente.", folder))
    return names_options[0][1] if names_options else {}


def issue(dataset: str, severity: str, source: Path, detail: str, root: Path) -> dict[str, str]:
    return {"dataset": dataset, "severidad": severity, "ruta": rel(source, root), "detalle": detail}


def add_issue(issues: list[dict], ds: str, sev: str, source: Path, detail: str, root: Path, limit=1000):
    if len(issues) < limit:
        issues.append(issue(ds, sev, source, detail, root))


def inspect_coco(dataset: str, root: Path, files: list[Path], image_index: dict[str, list[Path]], issues: list[dict]) -> dict:
    stats = {"archivos": 0, "imagenes_referenciadas": 0, "imagenes_no_encontradas": 0,
             "referencias_ambiguas": 0, "objetos": 0, "cajas": 0, "poligonos": 0, "rle": 0,
             "sin_geometria": 0, "cajas_invalidas": 0, "clases": Counter(), "nombres": {},
             "varios_objetos_en_imagen": 0}
    for p in files:
        if p.suffix.lower() != ".json":
            continue
        try:
            with p.open("r", encoding="utf-8-sig") as f:
                data = json.load(f)
        except (OSError, ValueError) as exc:
            add_issue(issues, dataset, "AVISO", p, f"JSON no legible: {exc}", root)
            continue
        if not (isinstance(data, dict) and isinstance(data.get("images"), list)
                and isinstance(data.get("annotations"), list)):
            continue
        stats["archivos"] += 1
        names = {str(c.get("id")): str(c.get("name", "SIN_NOMBRE"))
                 for c in data.get("categories", []) if isinstance(c, dict) and c.get("id") is not None}
        stats["nombres"].update(names)
        im_ref = {}
        count_image = Counter()
        for im in data["images"]:
            if not isinstance(im, dict) or im.get("id") is None:
                continue
            im_ref[str(im["id"])] = im
            stats["imagenes_referenciadas"] += 1
            ref = str(im.get("file_name", ""))
            hits = pick_images(ref, image_index, p, root) if ref else []
            if not hits:
                stats["imagenes_no_encontradas"] += 1
                add_issue(issues, dataset, "REVISAR", p, f"COCO imagen no encontrada: {ref}", root)
            elif len(hits) > 1:
                stats["referencias_ambiguas"] += 1
                add_issue(issues, dataset, "REVISAR", p, f"COCO referencia ambigua ({len(hits)}): {ref}", root)
        for ann in data["annotations"]:
            if not isinstance(ann, dict):
                continue
            stats["objetos"] += 1
            cls = str(ann.get("category_id", "SIN_ID"))
            stats["clases"][cls] += 1
            idimg = str(ann.get("image_id", "SIN_IMAGEN"))
            count_image[idimg] += 1
            im = im_ref.get(idimg, {})
            bb = ann.get("bbox")
            if isinstance(bb, (list, tuple)) and len(bb) == 4:
                stats["cajas"] += 1
                try:
                    x, y, w, h = map(float, bb)
                    iw, ih = im.get("width"), im.get("height")
                    bad = w <= 0 or h <= 0 or x < 0 or y < 0
                    if iw is not None and ih is not None:
                        bad |= x + w > float(iw) + 1 or y + h > float(ih) + 1
                    if bad:
                        stats["cajas_invalidas"] += 1
                except (ValueError, TypeError):
                    stats["cajas_invalidas"] += 1
            segmentation = ann.get("segmentation")
            if isinstance(segmentation, list) and any(isinstance(s, list) and len(s) >= 6 for s in segmentation):
                stats["poligonos"] += 1
            elif isinstance(segmentation, dict) and "counts" in segmentation:
                stats["rle"] += 1
            if not bb and not segmentation:
                stats["sin_geometria"] += 1
            if idimg not in im_ref:
                add_issue(issues, dataset, "REVISAR", p, f"Anotación COCO apunta a image_id ausente: {idimg}", root)
        stats["varios_objetos_en_imagen"] += sum(v >= 2 for v in count_image.values())
    return stats


def inspect_voc(dataset: str, root: Path, files: list[Path], image_index: dict[str, list[Path]], issues: list[dict]) -> dict:
    stats = {"archivos": 0, "objetos": 0, "cajas": 0, "cajas_invalidas": 0,
             "imagenes_no_encontradas": 0, "referencias_ambiguas": 0, "clases": Counter(),
             "varios_objetos_en_imagen": 0}
    stem_index = defaultdict(list)
    for paths in image_index.values():
        for im_path in paths:
            stem_index[im_path.stem.lower()].append(im_path)
    for p in files:
        if p.suffix.lower() != ".xml":
            continue
        try:
            root_xml = ET.parse(p).getroot()
        except (OSError, ET.ParseError) as exc:
            add_issue(issues, dataset, "AVISO", p, f"XML no legible: {exc}", root)
            continue
        if root_xml.tag.lower() != "annotation":
            continue
        stats["archivos"] += 1
        fname = root_xml.findtext("filename") or p.stem
        if not Path(fname).suffix:
            candidates = stem_index.get(fname.lower(), [])
        else:
            candidates = pick_images(fname, image_index, p, root)
        if not candidates:
            stats["imagenes_no_encontradas"] += 1
            add_issue(issues, dataset, "REVISAR", p, f"VOC imagen no encontrada: {fname}", root)
        elif len(candidates) > 1:
            stats["referencias_ambiguas"] += 1
        objs = root_xml.findall("object")
        if len(objs) >= 2:
            stats["varios_objetos_en_imagen"] += 1
        for obj in objs:
            stats["objetos"] += 1
            name = obj.findtext("name") or "SIN_NOMBRE"
            stats["clases"][name] += 1
            bb = obj.find("bndbox")
            if bb is not None:
                stats["cajas"] += 1
                try:
                    x1, y1, x2, y2 = [float(bb.findtext(k) or "nan") for k in ("xmin", "ymin", "xmax", "ymax")]
                    if not (x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0):
                        stats["cajas_invalidas"] += 1
                except ValueError:
                    stats["cajas_invalidas"] += 1
    return stats


def inspect_yolo(dataset: str, root: Path, files: list[Path], image_index: dict[str, list[Path]], issues: list[dict]) -> dict:
    labels = [p for p in files if is_yolo_label(p)]
    names = load_yolo_names(root, files, issues, dataset) if labels else {}
    stats = {"archivos": len(labels), "objetos": 0, "cajas": 0, "poligonos_posibles": 0,
             "otros_registros": 0, "filas_invalidas": 0, "cajas_invalidas": 0,
             "imagenes_no_encontradas": 0, "referencias_ambiguas": 0,
             "clases": Counter(), "nombres": {str(k): v for k, v in names.items()},
             "varios_objetos_en_imagen": 0, "etiquetas_vacias": 0}
    stem_index = defaultdict(list)
    for paths in image_index.values():
        for im_path in paths:
            stem_index[im_path.stem.lower()].append(im_path)
    for p in labels:
        # YOLO estándar: labels/train/foto.txt -> images/train/foto.jpg
        possible = list(stem_index.get(p.stem.lower(), []))
        if len(possible) > 1:
            s = split_name(relpath(p, root))
            if s != "sin_identificar":
                ps = [x for x in possible if split_name(relpath(x, root)) == s]
                if ps:
                    possible = ps
        if not possible:
            stats["imagenes_no_encontradas"] += 1
            add_issue(issues, dataset, "REVISAR", p, "Etiqueta YOLO sin imagen reconocible por nombre de archivo", root)
        elif len(possible) > 1:
            stats["referencias_ambiguas"] += 1
        try:
            rows = p.read_text(encoding="utf-8-sig").splitlines()
        except (OSError, UnicodeError) as exc:
            add_issue(issues, dataset, "REVISAR", p, f"Etiqueta YOLO no legible: {exc}", root)
            continue
        if not any(s.strip() for s in rows):
            stats["etiquetas_vacias"] += 1
        count = 0
        for line_idx, row in enumerate(rows, 1):
            values = row.strip().split()
            if not values:
                continue
            try:
                class_id = int(values[0])
                coords = [float(v) for v in values[1:]]
                if class_id < 0:
                    raise ValueError("ID negativo")
                if not all(float("-inf") < v < float("inf") for v in coords):
                    raise ValueError("Coordenadas no finitas")
            except ValueError:
                stats["filas_invalidas"] += 1
                add_issue(issues, dataset, "REVISAR", p, f"YOLO fila {line_idx} inválida", root)
                continue
            if len(coords) == 4:
                stats["cajas"] += 1
                x, y, w, h = coords
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1
                        and x - w / 2 >= -1e-5 and x + w / 2 <= 1.00001
                        and y - h / 2 >= -1e-5 and y + h / 2 <= 1.00001):
                    stats["cajas_invalidas"] += 1
            elif len(coords) >= 6 and len(coords) % 2 == 0:
                # Distintas variantes YOLO pueden usar coordenadas tipo polígono u OBB.
                stats["poligonos_posibles"] += 1
                if any(v < 0 or v > 1 for v in coords):
                    stats["filas_invalidas"] += 1
            else:
                stats["otros_registros"] += 1
            key = str(class_id)
            stats["clases"][key] += 1
            stats["objetos"] += 1
            count += 1
            if names and class_id not in names:
                add_issue(issues, dataset, "REVISAR", p, f"ID de clase YOLO {class_id} ausente en data.yaml", root)
        if count >= 2:
            stats["varios_objetos_en_imagen"] += 1
    return stats


def inspect_images(ds: str, root: Path, paths: list[Path], n: int, all_images: bool, issues: list[dict]) -> tuple[dict, list[dict]]:
    if all_images:
        sample = paths
    elif len(paths) >= 2 and n >= 2:
        sample_count = min(n, len(paths))
        indices = sorted({round(j * (len(paths) - 1) / (sample_count - 1))
                          for j in range(sample_count)})
        sample = [paths[i] for i in indices]
    else:
        sample = paths[:max(0, n)]
    meta = {"analizadas": len(sample), "errores": 0, "modos": Counter(),
            "anchos": [], "altos": [], "extensiones": Counter(p.suffix.lower() for p in paths)}
    rows = []
    for p in sample:
        row = {"dataset": ds, "ruta": rel(p, root), "split": split_name(relpath(p, root)),
               "extension": p.suffix.lower(), "ancho": "", "alto": "", "modo": "", "estado": "OK"}
        try:
            with Image.open(p) as im:
                row["ancho"], row["alto"], row["modo"] = im.width, im.height, im.mode
                meta["modos"][im.mode] += 1
                meta["anchos"].append(im.width)
                meta["altos"].append(im.height)
                im.verify()
        except (OSError, ValueError, SyntaxError) as exc:
            row["estado"] = f"ERROR: {str(exc)[:160]}"
            meta["errores"] += 1
            add_issue(issues, ds, "REVISAR", p, f"No se pudo verificar imagen: {exc}", root)
        rows.append(row)
    return meta, rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def duplicates_by_hash(by_ds: dict[str, list[Path]], root: Path, issues: list[dict]) -> dict:
    by_size: dict[int, list[tuple[str, Path]]] = defaultdict(list)
    for ds, files in by_ds.items():
        for p in files:
            try:
                by_size[p.stat().st_size].append((ds, p))
            except OSError as exc:
                add_issue(issues, ds, "AVISO", p, f"No se pudo leer tamaño: {exc}", root)
    by_hash: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    hashed = 0
    for group in by_size.values():
        if len(group) < 2:
            continue
        for ds, p in group:
            try:
                by_hash[sha256_file(p)].append((ds, p))
                hashed += 1
            except OSError as exc:
                add_issue(issues, ds, "AVISO", p, f"No se pudo calcular hash: {exc}", root)
    duplicates = [items for items in by_hash.values() if len(items) > 1]
    cross_ds = 0
    cross_split = 0
    examples = []
    for items in duplicates:
        has_cross_ds = len({ds for ds, _ in items}) > 1
        partitions = {split_name(relpath(p, root)) for _, p in items}
        has_cross_split = len(partitions - {"sin_identificar"}) > 1
        cross_ds += has_cross_ds
        cross_split += has_cross_split
        if len(examples) < 30:
            examples.append({"archivos": [{"dataset": ds, "ruta": rel(p, root)} for ds, p in items[:8]],
                             "entre_datasets": has_cross_ds, "entre_splits": has_cross_split})
    return {"comprobado": True, "imagenes_hasheadas": hashed, "grupos_duplicados": len(duplicates),
            "grupos_entre_datasets": cross_ds, "grupos_entre_splits": cross_split, "ejemplos": examples}


def analyze_dataset(ds: str, root: Path, n_sample: int, all_images: bool, issues: list[dict]) -> tuple[dict, list[dict], list[Path]]:
    if not root.is_dir():
        add_issue(issues, ds, "BLOQUEANTE", root, "No se encuentra el directorio del dataset", root.parent)
        return {"nombre": ds, "ruta": str(root), "existe": False}, [], []
    all_files = [p for p in root.rglob("*") if p.is_file()]
    images = sorted((p for p in all_files if p.suffix.lower() in IMAGE_EXTS and not detect_mask(p)),
                    key=lambda x: str(x).lower())
    masks = [p for p in all_files if p.suffix.lower() in IMAGE_EXTS and detect_mask(p)]
    name_index: dict[str, list[Path]] = defaultdict(list)
    for p in images:
        name_index[p.name.lower()].append(p)
    im_stats, im_rows = inspect_images(ds, root, images, n_sample, all_images, issues)
    coco = inspect_coco(ds, root, all_files, name_index, issues)
    voc = inspect_voc(ds, root, all_files, name_index, issues)
    yolo = inspect_yolo(ds, root, all_files, name_index, issues)
    partitions = Counter(split_name(relpath(p, root)) for p in images)
    # Informamos el total de archivos por tipos; no confundimos máscaras con imágenes de entrada.
    result = {"nombre": ds, "ruta": root.name, "existe": True,
              "archivos_totales": len(all_files), "imagenes": len(images), "mascaras_archivos": len(masks),
              "mascaras_ejemplos": [rel(p, root) for p in masks[:8]],
              "imagenes_por_split": dict(partitions), "imagenes_por_extension": dict(im_stats["extensiones"]),
              "muestra_inspeccionada": im_stats["analizadas"], "imagenes_error_muestra": im_stats["errores"],
              "modos_muestra": dict(im_stats["modos"]),
              "dimensiones_muestra": {
                  "ancho_min": min(im_stats["anchos"], default=None),
                  "ancho_max": max(im_stats["anchos"], default=None),
                  "alto_min": min(im_stats["altos"], default=None),
                  "alto_max": max(im_stats["altos"], default=None)},
              "anotaciones": {"coco": coco, "voc": voc, "yolo": yolo},
              "ejemplos_imagenes": [rel(p, root) for p in images[:5]],
              "ejemplos_anotaciones": [rel(p, root) for p in all_files
                                      if p.suffix.lower() in {".xml", ".json"} or is_yolo_label(p)][:8],
              "posibles_splits_voc": sum("imagesets" in [x.lower() for x in p.parts]
                                        for p in all_files if p.suffix.lower() == ".txt")}
    return result, im_rows, images


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def top_classes(source: dict, n=18) -> list[tuple[str, str, int]]:
    names = source.get("nombres", {})
    return [(k, names.get(str(k), k), v) for k, v in source.get("clases", {}).most_common(n)]


def make_markdown(results: list[dict], duplicates: dict, issues: list[dict], all_images: bool, sample: int) -> str:
    lines = ["# Informe preliminar de auditoría de datasets IMO", "",
             f"Fecha de auditoría: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (hora del equipo)", "",
             "**Estado:** inventario estructural automático. La compatibilidad semántica definitiva necesita revisar ejemplos visuales, definiciones de clases, procedencia y licencias.", "",
             "## Inventario", "",
             "| Dataset | Imágenes | COCO (archivos/objetos) | VOC (archivos/objetos) | YOLO (etiquetas/objetos) | Archivos de máscara |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in results:
        if not r.get("existe"):
            lines.append(f"| {r['nombre']} | NO ENCONTRADO | — | — | — | — |")
            continue
        a = r["anotaciones"]
        lines.append(f"| {r['nombre']} | {r['imagenes']} | {a['coco']['archivos']} / {a['coco']['objetos']} | {a['voc']['archivos']} / {a['voc']['objetos']} | {a['yolo']['archivos']} / {a['yolo']['objetos']} | {r['mascaras_archivos']} |")
    lines += ["", "## Hallazgos por dataset", ""]
    for r in results:
        lines.append(f"### {r['nombre']}")
        if not r.get("existe"):
            lines += ["Carpeta no encontrada. Revisa `config/datasets.json`.", ""]
            continue
        a = r["anotaciones"]
        lines += [f"- Extensiones de imágenes: {r['imagenes_por_extension']}.",
                  f"- Imágenes por partición inferida del nombre de carpeta: {r['imagenes_por_split']}.",
                  f"- Imágenes verificadas con Pillow: {r['muestra_inspeccionada']} de {r['imagenes']} (con errores: {r['imagenes_error_muestra']}).",
                  f"- Dimensiones observadas en esa muestra: {r['dimensiones_muestra']}; modos: {r['modos_muestra']}.",
                  f"- Archivos de máscara encontrados por nombres de carpeta: {r['mascaras_archivos']}; su semántica NO está validada.",
                  f"- COCO: {a['coco']['cajas']} cajas, {a['coco']['poligonos']} anotaciones con polígonos, {a['coco']['rle']} con RLE; {a['coco']['imagenes_no_encontradas']} referencias a imágenes no resueltas; {a['coco']['referencias_ambiguas']} ambiguas; {a['coco']['cajas_invalidas']} cajas inválidas.",
                  f"- Pascal VOC: {a['voc']['cajas']} cajas, {a['voc']['imagenes_no_encontradas']} imágenes no resueltas; {a['voc']['cajas_invalidas']} cajas inválidas. Listas de ImageSets detectadas: {r['posibles_splits_voc']}.",
                  f"- YOLO: {a['yolo']['cajas']} cajas, {a['yolo']['poligonos_posibles']} polígonos/OBB posibles (verificación requerida), {a['yolo']['otros_registros']} geometrías no reconocidas, {a['yolo']['filas_invalidas']} filas inválidas, {a['yolo']['imagenes_no_encontradas']} imágenes no resueltas, {a['yolo']['referencias_ambiguas']} ambiguas, {a['yolo']['cajas_invalidas']} cajas fuera de rango.",
                  f"- Casos de varias anotaciones por imagen: COCO {a['coco']['varios_objetos_en_imagen']}, VOC {a['voc']['varios_objetos_en_imagen']}, YOLO {a['yolo']['varios_objetos_en_imagen']}. Estas cifras pueden contar la misma imagen en distintos formatos.",
                  "", "**Clases que aparecen en las anotaciones:**", ""]
        for fmt in ("coco", "voc", "yolo"):
            objs = top_classes(a[fmt])
            if objs:
                lines += [f"Formato **{fmt.upper()}**:", "", "| ID | Clase (según fuente) | Instancias |", "|---|---|---:|"]
                lines.extend(f"| {key} | {name.replace('|', '/')} | {count} |" for key, name, count in objs)
                if len(a[fmt]["clases"]) > len(objs):
                    lines.append(f"| … | Ver todas en `clases.csv` | {len(a[fmt]['clases']) - len(objs)} clases adicionales |")
                lines.append("")
        if not any(a[k]["archivos"] for k in ("coco", "voc", "yolo")):
            lines += ["**No se detectó un formato de anotación reconocido.** No inferir que las imágenes están etiquetadas.", ""]
        lines += [f"- Ejemplos de imágenes: {', '.join('`' + p + '`' for p in r['ejemplos_imagenes']) or '(ninguna)'}.",
                  f"- Ejemplos de archivos de anotación: {', '.join('`' + p + '`' for p in r['ejemplos_anotaciones']) or '(ninguno)'}.", ""]
    lines += ["## Compatibilidad: criterios antes de unificar", "",
              "| Criterio | Qué sabemos | Qué falta |", "|---|---|---|",
              "| Técnica | Se detectaron formatos, cajas, polígonos posibles y archivos de máscara en el inventario. | Definir formato objetivo y validar conversiones exactas con ejemplos. |",
              "| Semántica | Los nombres e IDs de clases se muestran literalmente en las tablas. | Verificar si cada clase representa una etiqueta IMO o un objeto distinto; definir ontología. |",
              "| Geométrica | Una caja describe ubicación aproximada, NO una máscara de píxeles. | Verificar máscaras reales o definir un plan de anotación si se exige segmentación. |",
              "| Estadística | Hay una distribución inicial de anotaciones y particiones. | Separar splits, evaluar desbalance, duplicados y cambio de dominio. |",
              "| Legal | No se inspeccionó por completo licencia ni procedencia. | Verificar licencias y restricciones de uso y publicación. |", ""]
    lines.append("## Duplicados")
    if duplicates.get("comprobado"):
        lines += [f"Se calcularon SHA256 para {duplicates['imagenes_hasheadas']} imágenes candidatas por tamaño; se identificaron {duplicates['grupos_duplicados']} grupos de imágenes binarias idénticas, {duplicates['grupos_entre_datasets']} entre datasets y {duplicates['grupos_entre_splits']} entre splits.", ""]
        for ex in duplicates.get("ejemplos", [])[:8]:
            lines.append("- " + "; ".join(f"{x['dataset']}: `{x['ruta']}`" for x in ex["archivos"][:4]))
    else:
        lines.append("**No se ejecutó comprobación por hash.** Ejecutar `--hash-duplicados` antes de determinar splits o mezclar datasets.")
    lines += ["", "## Incidencias", "", f"Total de incidencias registradas: **{len(issues)}** (puede haber límite de registro).", ""]
    for i in issues[:25]:
        lines.append(f"- **{i['severidad']}** · {i['dataset']} · `{i['ruta']}` · {i['detalle']}")
    if len(issues) > 25:
        lines.append(f"- Se omiten {len(issues) - 25} del informe; ver `incidencias.csv`.")
    lines += ["", "## Decisiones siguientes (requieren aprobación)", "",
              "1. Revisar visualmente ejemplos de cada clase y la correspondencia real imagen-anotación.",
              "2. Confirmar si el problema será detección, segmentación real, clasificación o pipeline de varias etapas.",
              "3. Diseñar una ontología que distinga IMO/IMDG de clases genéricas o elementos del contenedor.",
              "4. Aprobar un mapeo explícito entre IDs de clases de origen y clases canónicas; no forzar equivalencias.",
              "5. Elegir formatos y particiones finales sin modificar datos originales, previa revisión de licencias y duplicados.",
              "6. Solo después: elaborar y probar código de uniformización/EDA y líneas base.", "",
              f"**Cobertura de imágenes:** {'100% solicitado' if all_images else f'muestra de hasta {sample} por dataset'}; las cifras de tamaños/resolución y corrupción se refieren únicamente a las imágenes inspeccionadas."
              ]
    return "\n".join(lines) + "\n"


def to_serializable(obj: Any) -> Any:
    if isinstance(obj, Counter):
        return dict(obj)
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    return obj


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Auditoría no destructiva de datasets IMO")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent,
                        help="Carpeta DEEP LEARNING que contiene los tres datasets")
    parser.add_argument("--muestra-imagenes", type=int, default=250,
                        help="Máximo de imágenes a verificar por dataset (por defecto 250)")
    parser.add_argument("--todas-imagenes", action="store_true", help="Verificar todas las imágenes (puede tardar)")
    parser.add_argument("--hash-duplicados", action="store_true", help="Comprobar hashes para candidatos con mismo tamaño")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.muestra_imagenes < 0:
        parser.error("--muestra-imagenes debe ser >= 0")
    cfg_file = root / "config" / "datasets.json"
    if not cfg_file.exists():
        parser.error(f"No existe configuración: {cfg_file}")
    with cfg_file.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    output = (root / cfg.get("directorio_resultados", "resultados/auditoria")).resolve()
    if not output.is_relative_to(root):
        parser.error("La salida debe ubicarse dentro de la raíz del proyecto")
    source_roots = [(d["nombre"], (root / d["ruta"]).resolve()) for d in cfg["datasets"]]
    for name, p in source_roots:
        if not p.is_relative_to(root) or (output == p or p in output.parents or output in p.parents):
            parser.error(f"Rutas no seguras o solapadas para dataset {name}: {p}")
    output.mkdir(parents=True, exist_ok=True)
    issues: list[dict] = []
    results, image_rows = [], []
    all_img_by_ds = {}
    for name, path in source_roots:
        print(f"[AUDITORÍA] {name}: {path}", flush=True)
        r, rows, paths = analyze_dataset(name, path, args.muestra_imagenes, args.todas_imagenes, issues)
        results.append(r)
        image_rows.extend(rows)
        all_img_by_ds[name] = paths
    duplicates = ({"comprobado": False} if not args.hash_duplicados
                  else duplicates_by_hash(all_img_by_ds, root, issues))
    classes = []
    for result in results:
        if not result.get("existe"):
            continue
        for fmt, source in result["anotaciones"].items():
            for class_id, n in source["clases"].most_common():
                classes.append({"dataset": result["nombre"], "formato_anotacion": fmt.upper(),
                                "clase_id_original": class_id,
                                "nombre_clase_original": source.get("nombres", {}).get(str(class_id), class_id),
                                "numero_anotaciones": n})
    write_csv(output / "clases.csv", classes,
              ["dataset", "formato_anotacion", "clase_id_original", "nombre_clase_original", "numero_anotaciones"])
    write_csv(output / "muestra_imagenes.csv", image_rows,
              ["dataset", "ruta", "split", "extension", "ancho", "alto", "modo", "estado"])
    write_csv(output / "incidencias.csv", issues, ["dataset", "severidad", "ruta", "detalle"])
    summary = {"ejecutado_en": datetime.now().isoformat(timespec="seconds"),
               "raiz": str(root), "parametros": {"muestra_imagenes": args.muestra_imagenes,
               "todas_imagenes": args.todas_imagenes, "hash_duplicados": args.hash_duplicados},
               "datasets": results, "duplicados": duplicates, "incidencias_registradas": len(issues)}
    with (output / "resumen.json").open("w", encoding="utf-8") as f:
        json.dump(to_serializable(summary), f, ensure_ascii=False, indent=2)
    (output / "INFORME_AUDITORIA.md").write_text(
        make_markdown(results, duplicates, issues, args.todas_imagenes, args.muestra_imagenes), encoding="utf-8")
    print(f"\n[LISTO] Auditoría escrita en: {output}")
    print("No se han modificado las carpetas originales. Lee INFORME_AUDITORIA.md.")
    missing = [r["nombre"] for r in results if not r.get("existe")]
    if missing:
        print("[ATENCIÓN] No se localizaron estas carpetas: " + ", ".join(missing))
        print("Comprueba que hayas extraído el paquete EN la carpeta DEEP LEARNING, no dentro de otra carpeta.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
