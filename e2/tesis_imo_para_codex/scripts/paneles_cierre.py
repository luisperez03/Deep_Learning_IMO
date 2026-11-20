"""Genera únicamente las nueve predicciones cualitativas preseleccionadas (CPU)."""
import csv
import io
import json
import zipfile

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from dataset_ready import load_example, metrics
from inicio_voc import PROJECT, sha
from modelos_base import MODELS
from preparar_base import letterbox, read_csv

CASOS = [("image_1118", "Alto"), ("image_1241", "Central"), ("image_1656", "Bajo")]
REDES = {"fcn": "FCN", "segnet": "SegNet", "unet": "U-Net"}
OUT = PROJECT / "libro" / "figuras"


def lienzo(columnas):
    canvas = Image.new("RGB", (18 + 268 * len(columnas), 936), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=19)
    for i, name in enumerate(columnas):
        draw.text((16 + 268*i, 10), name, font=font, fill="#16394a")
    for i, (ident, nivel) in enumerate(CASOS):
        draw.text((16, 43 + i*300), f"{nivel}  /  {ident}", font=font, fill="#16394a")
    return canvas


def poner(canvas, fila, col, imagen):
    canvas.paste(imagen, (16 + 268*col, 72 + fila*300))


def borde(mask):
    im = Image.fromarray(mask.astype("uint8")*255)
    return mask & (np.asarray(im.filter(ImageFilter.MinFilter(3))) == 0)


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    with zipfile.ZipFile(PROJECT/"IMO_test_final.zip") as archive:
        meta = json.loads(archive.read("procedencia.json"))
        saved = list(csv.DictReader(io.StringIO(archive.read("por_imagen.csv").decode())))
    prep = PROJECT/"resultados/preentrenamiento"
    for file in ["test.csv","normalization_train.json"]:
        assert sha(prep/file) == meta["frozen_files"][file]
    test = {r["id"]:r for r in read_csv(prep/"test.csv")}
    models = {}
    for ck in meta["checkpoints"]:
        if ck["seed"] != 11:
            continue
        path = PROJECT/ck["file"]
        assert sha(path) == ck["sha256"]
        model = MODELS[ck["model"]]()
        model.load_state_dict(torch.load(path, map_location="cpu",weights_only=True))
        models[ck["model"]] = model.eval()
    comparison = lienzo(["Imagen RGB", "Referencia", *REDES.values()])
    errors = lienzo(list(REDES.values()))
    contours = lienzo(list(REDES.values()))
    lines = ["| Caso | Red | VP | FP | FN | Dice CPU | Dice del ZIP |",
             "|---|---|---:|---:|---:|---:|---:|"]
    deviations = []
    for i,(ident,_) in enumerate(CASOS):
        row = test[ident]
        image = PROJECT.parent/row["image_path"]
        assert sha(image)==row["image_sha256"] and sha(PROJECT/row["binary_mask_path"])==row["binary_mask_sha256"]
        x,y,valid = load_example(row)
        target = (y==1) & valid
        with Image.open(image) as src:
            photo,_ = letterbox(src.convert("RGB"),256)
        poner(comparison,i,0,photo)
        poner(comparison,i,1,Image.fromarray(target.astype("uint8")*255))
        batch = torch.from_numpy(x.copy())[None]
        for j,(name,label) in enumerate(REDES.items()):
            prob = models[name](batch)[0,0].sigmoid().numpy()
            assert np.isfinite(prob).all()
            pred = (prob>=0.5) & valid
            measured = metrics(prob,target,valid)
            record = next(r for r in saved if (r["id"],r["model"],r["seed"])==(ident,name,"11"))
            deviations.extend(abs(v-float(record[k])) for k,v in measured.items())
            hit,fp,fn = pred & target, pred & ~target, target & ~pred
            lines.append(f"| {ident} | {label} | {hit.sum()} | {fp.sum()} | {fn.sum()} | "
                         f"{measured['Dice']:.5f} | {float(record['Dice']):.5f} |")
            poner(comparison,i,j+2,Image.fromarray(pred.astype("uint8")*255))
            base = np.asarray(photo).copy()
            tinted = (base.astype(float)*0.50).astype("uint8")
            tinted[hit],tinted[fp],tinted[fn] = (90,182,126),(255,144,39),(89,177,255)
            poner(errors,i,j,Image.fromarray(tinted))
            bounds = base.copy()
            # Solo transiciones entre píxeles válidos, sin bordes artificiales de relleno.
            yy,xx = np.where(valid)
            window = np.s_[yy.min():yy.max()+1,xx.min():xx.max()+1]
            truth_edge,pred_edge = borde(target[window]),borde(pred[window])
            crop = bounds[window]
            crop[truth_edge],crop[pred_edge],crop[truth_edge & pred_edge] = (89,177,255),(255,144,39),(255,255,255)
            poner(contours,i,j,Image.fromarray(bounds))
    OUT.mkdir(exist_ok=True)
    for canvas,file in [(comparison,"comparacion"),(errors,"errores"),(contours,"contornos")]:
        canvas.save(OUT/(file+".png"))
    lines += ["", f"Diferencia absoluta máxima CPU frente a T4 sobre las cuatro métricas de los nueve ejemplos: {max(deviations):.9g}. El cotejo es por métricas; el ZIP no contiene mapas T4 para certificar identidad píxel a píxel."]
    (PROJECT/"libro/conteos_panel.inc").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
