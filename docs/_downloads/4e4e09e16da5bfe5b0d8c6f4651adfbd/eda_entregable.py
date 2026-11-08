"""Complementa el EDA histórico sin ajustar ninguna transformación."""
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage

from analisis_entregable import BOOK, ROOT, fmt, table
from preparar_base import letterbox


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    prep = ROOT/"resultados/preentrenamiento"
    initial = ROOT/"resultados/inicio_voc"
    assets = BOOK/"figuras"
    assets.mkdir(exist_ok=True)
    for filename in ["08_balance_pixeles.png","07_objetos_pequenos.png","11_similitud.png","05_aspecto.png"]:
        shutil.copyfile(initial/"graficos"/filename, assets/filename)
    shutil.copyfile(ROOT/"resultados/revision_voc/mascara_image_1461.png",assets/"revision_1461.png")
    shutil.copyfile(ROOT/"notebooks/01_eda_voc.ipynb", BOOK/"eda_original.ipynb")
    counts = json.loads((initial/"resumen.json").read_text())["pixel_counts_valid_pairs"]
    fg = 1-counts["background"]/sum(counts.values())
    entropy = -fg*np.log2(fg)-(1-fg)*np.log2(1-fg)
    inventory = {r["id"]: r for r in read(initial/"inventario.csv")}
    features = {r["id"]: r for r in read(ROOT/"resultados/revision_voc/caracteristicas_imagen.csv")}
    rng = np.random.default_rng(20261107)
    rows = []
    brightness = []
    memberships = []
    train = read(prep/"train.csv")
    for part in ["train","val","test"]:
        records = read(prep/f"{part}.csv")
        groups = defaultdict(list)
        for r in records:
            groups[r["group_id"]].append(float(features[r["id"]]["brightness_thumb"]))
        memberships.append(set(groups))
        brightness.append(np.array([np.mean(v) for v in groups.values()]))
        median = lambda field: np.median([float(inventory[r["id"]][field]) for r in records])
        rows.append([part,len(records),len(groups),fmt(median("foreground_fraction")),int(median("width")),
                     int(median("height")),fmt(brightness[-1].mean())])
    assert sum(map(len,memberships)) == len(set.union(*memberships))
    pooled = np.concatenate(brightness)
    sizes = [len(b) for b in brightness]
    def dispersion(samples):
        return sum(len(a)*(a.mean()-pooled.mean())**2 for a in samples)
    observed = dispersion(brightness)
    bounds = np.cumsum(sizes)[:-1]
    hits = sum(dispersion(np.split(rng.permutation(pooled),bounds)) >= observed for _ in range(4999))
    hist = np.zeros((2,3,32),dtype=np.int64)
    areas, numbers, shapes = [], [], []
    for row in train:
        with Image.open(ROOT.parent/row["image_path"]) as source:
            image, valid = letterbox(source.convert("RGB"),256)
        with Image.open(ROOT/row["binary_mask_path"]) as source:
            mask, _ = letterbox(source.convert("L"),256,True)
        y = np.asarray(mask) == 1
        rgb = np.asarray(image)
        labels,n = ndimage.label(y,structure=np.ones((3,3)))
        numbers.append(n)
        for region in ndimage.find_objects(labels):
            h,w = [axis.stop-axis.start for axis in region]
            shapes.append(max(h,w)/min(h,w))
        areas.extend(np.bincount(labels.ravel())[1:])
        for category in [0,1]:
            pixels = rgb[valid & (y==category)]
            for channel in range(3):
                hist[category,channel] += np.bincount(pixels[:,channel]//8,minlength=32)
    fig,axes=plt.subplots(1,2,figsize=(8,3.3),constrained_layout=True,sharey=True)
    for category,ax in enumerate(axes):
        for channel,(label,color) in enumerate(zip(["R","G","B"],["#a74439","#237759","#355c9c"])):
            h=hist[category,channel]
            ax.plot(np.arange(4,256,8),h/h.sum(),color=color,label=label)
        ax.set(title=["Fondo","Etiqueta"][category],xlabel="Intensidad RGB (0–255)")
        ax.legend()
    axes[0].set_ylabel("Proporción por intervalo")
    fig.savefig(assets/"rgb_train.png",dpi=155)
    plt.close(fig)
    lines = [f"En los 1.583 pares originales, la etiqueta ocupa {100*fg:.2f} % de los píxeles y el fondo {100*(1-fg):.2f} %. La razón fondo/etiqueta es {(1-fg)/fg:.1f} y la entropía binaria es {entropy:.3f} bits por píxel.",
             "## Regiones que llegan al modelo",
             f"En las 1.015 máscaras de entrenamiento a 256 × 256, la mediana de componentes por imagen es {np.median(numbers):.0f}. Hay {len(areas)} regiones de ocho vecinos. Los percentiles 10, 50 y 90 de área son {', '.join(str(round(float(x),1)) for x in np.percentile(areas,[10,50,90]))} píxeles. La razón mediana entre el lado largo y el corto de su caja envolvente es {np.median(shapes):.2f}. Una componente puede reunir rótulos en contacto o dividir uno ocluido. El área total de etiqueta nunca baja de 16 píxeles en las máscaras de entrenamiento a esta resolución, pero componentes individuales pueden ser menores. Esto sustenta conservar el encuadre y medir solapamiento de la clase pequeña.",
             "![Histogramas de RGB por clase en entrenamiento](figuras/rgb_train.png)",
             "Las intensidades de etiqueta y fondo se solapan en los tres canales. Un umbral de color único perdería parte de ambas distribuciones. Para normalizar se usaron exclusivamente píxeles válidos de train, con medias RGB 0,604/0,561/0,507 y desviaciones 0,233/0,227/0,240.",
             "## Comparación de las particiones congeladas", table(["Partición","Imágenes","Grupos","Mediana fracción etiqueta","Ancho mediano","Alto mediano","Brillo medio por grupo"],rows),
             f"El test tiene menor fracción mediana de etiqueta que train. Para el brillo de miniaturas se compararon las medias de grupo entre las tres particiones. El estadístico es la suma de desviaciones cuadráticas de cada media de partición respecto de la global, ponderadas por el número de grupos. Una permutación exploratoria de etiquetas de partición entre grupos, conservando sus tamaños en número de grupos (4.999 permutaciones, semilla 20261107), da p = {fmt((hits+1)/5000)}. El diagnóstico cubre un atributo y presupone intercambiabilidad de grupos. Ni un brillo similar ni el resultado de esta prueba certifican igualdad de dominio. Se conservaron las particiones originales del experimento."]
    (BOOK/"eda_calculos.inc").write_text("\n\n".join(lines)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
