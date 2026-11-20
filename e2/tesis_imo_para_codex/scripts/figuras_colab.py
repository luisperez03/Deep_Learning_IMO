"""Inferencia cualitativa sobre tres imágenes predeterminadas. Solo para Colab."""
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from dataset_ready import load_example
from modelos_base import MODELS
from preparar_base import PROJECT, letterbox
from inicio_voc import sha


@torch.inference_mode()
def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Selecciona un kernel de Colab con GPU.")
    with (PROJECT/"casos.csv").open(encoding="utf-8-sig",newline="") as source:
        rows = list(csv.DictReader(source))
    assert [r["id"] for r in rows] == ["image_1656","image_1241","image_1118"]
    expected = json.loads((PROJECT/"huellas.json").read_text())
    for name,digest in expected.items():
        if sha(PROJECT/name) != digest:
            raise ValueError(f"Huella diferente: {name}")
    models = {}
    for name,constructor in MODELS.items():
        model = constructor()
        model.load_state_dict(torch.load(PROJECT/f"{name}.pt",map_location="cpu",weights_only=True))
        models[name] = model.to("cuda").eval()
    fig,axes = plt.subplots(3,5,figsize=(12,8),constrained_layout=True)
    out = Path("/content/IMO_figuras")
    out.mkdir(exist_ok=True)
    for row,panel in zip(rows,axes):
        x,y,valid = load_example(row)
        image_path = PROJECT.parent/row["image_path"]
        with Image.open(image_path) as source:
            photo,_ = letterbox(source.convert("RGB"),256)
        panel[0].imshow(photo)
        panel[0].set_title(row["id"])
        panel[1].imshow(y*valid,cmap="gray",vmin=0,vmax=1)
        panel[1].set_title("Anotación")
        tensor = torch.from_numpy(x.copy())[None].cuda()
        for ax,(name,model) in zip(panel[2:],models.items()):
            logits = model(tensor)[0,0]
            if not torch.isfinite(logits).all():
                raise ValueError("Salida no finita")
            pred = (logits.sigmoid().cpu().numpy() >= .5) & valid
            Image.fromarray((pred*255).astype("uint8")).save(out/f"{row['id']}_{name}.png")
            ax.imshow(pred,cmap="gray",vmin=0,vmax=1)
            ax.set_title(name.upper().replace("UNET","U-Net").replace("SEGNET","SegNet"))
        for ax in panel:
            ax.axis("off")
    fig.savefig(out/"comparacion.png",dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
