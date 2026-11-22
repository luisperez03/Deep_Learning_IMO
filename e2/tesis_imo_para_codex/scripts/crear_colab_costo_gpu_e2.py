"""Genera colab/Medir_Costo_GPU_E2.ipynb a partir de las celdas de resultados/costo_e2/COLAB_MEDIR_COSTO.md.

  python scripts/crear_colab_costo_gpu_e2.py

Fuente única: los bloques ```python del Markdown, en orden; el texto previo a la primera celda va como celda de
texto. Así el cuaderno y la guía no divergen (filas 44 del checklist: VM-UNet solo corre en GPU).
"""
import json
import re
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
GUIA = RAIZ / "resultados" / "costo_e2" / "COLAB_MEDIR_COSTO.md"
SALIDA = RAIZ / "colab" / "Medir_Costo_GPU_E2.ipynb"


def celda(tipo, texto):
    fuente = [l + "\n" for l in texto.rstrip("\n").split("\n")]
    fuente[-1] = fuente[-1].rstrip("\n")
    c = {"cell_type": tipo, "metadata": {}, "source": fuente}
    if tipo == "code":
        c.update(execution_count=None, outputs=[])
    return c


def main():
    md = GUIA.read_text(encoding="utf-8")
    bloques = re.findall(r"```python\n(.*?)```", md, flags=re.S)
    if len(bloques) != 4:
        raise SystemExit(f"Se esperaban 4 celdas en {GUIA}, hay {len(bloques)}")
    intro = md.split("## Celdas")[0].strip()
    nb = {"cells": [celda("markdown", intro)] + [celda("code", b) for b in bloques],
          "metadata": {"accelerator": "GPU", "colab": {"gpuType": "L4", "provenance": []},
                       "kernelspec": {"display_name": "Python 3", "name": "python3"}},
          "nbformat": 4, "nbformat_minor": 0}
    SALIDA.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{SALIDA} con {len(bloques)} celdas de código")


if __name__ == "__main__":
    main()
