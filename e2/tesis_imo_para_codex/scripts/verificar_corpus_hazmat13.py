"""Verifica que una copia local de HAZMAT-13 (formato VOC) coincide con la descrita en docs/DATA_CARD_HAZMAT13.md.

  python scripts/verificar_corpus_hazmat13.py RUTA/VOC2012

No descarga nada ni escribe en el corpus: solo lee. Comprueba conteos, clases, correspondencia entre imágenes,
máscaras y XML, y dos huellas agregadas (SHA256 sobre los nombres y el SHA256 de cada archivo, en orden
alfabético). Termina con código 1 si algo no coincide. Las huellas esperadas son las de la copia usada en el
proyecto; una copia descargada de nuevo desde el Drive de los autores debe dar las mismas si es la misma versión.
"""
import hashlib
import re
import sys
from collections import Counter
from pathlib import Path

ESPERADO = {
    "imagenes_jpg": 1685,
    "mascaras_png": 1583,
    "xml_cajas": 1582,
    "imagenes_sin_mascara": 102,
    "mascaras_sin_xml": ["image_1018"],
    "cajas_totales": 2639,
    "carpetas_de_anotador": 7,
    "huella_jpg": "58b938d42de5f4fb514d0b734554fc4a4ec6463e3840d959f995e0f2aeaf9f3b",
    "huella_xml": "090ab361d20d38e3d9191614c30581d48a70771e40bc03090d7925015996518f",
    "cajas_por_clase": {
        "poison": 79, "oxygen": 256, "flammable": 174, "flammable-solid": 216, "corrosive": 152, "dangerous": 203,
        "non-flammable-gas": 185, "organic-peroxide": 163, "explosive": 406, "radioactive": 281,
        "inhalation-hazard": 250, "spontaneously-combustible": 125, "infectious-substance": 149,
    },
}


def huella(carpeta: Path) -> str:
    """SHA256 sobre (nombre, SHA256 del contenido) de cada archivo en orden alfabético."""
    h = hashlib.sha256()
    for f in sorted(carpeta.iterdir()):
        h.update(f.name.encode())
        h.update(hashlib.sha256(f.read_bytes()).digest())
    return h.hexdigest()


def huella_xml(carpeta: Path) -> str:
    """SHA256 sobre (nombre, bytes) de cada XML en orden alfabético."""
    h = hashlib.sha256()
    for f in sorted(carpeta.iterdir()):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    raiz = Path(sys.argv[1])
    jpg, png, xml = raiz / "JPEGImages", raiz / "SegmentationClass", raiz / "bboxes" / "annotations"
    for p in (jpg, png, xml):
        if not p.is_dir():
            print(f"Falta la carpeta {p}")
            return 2
    ids_jpg = {f.stem for f in jpg.iterdir()}
    ids_png = {f.stem for f in png.iterdir()}
    ids_xml = {f.stem for f in xml.iterdir()}
    cajas, carpetas = Counter(), set()
    for f in xml.iterdir():
        t = f.read_text(encoding="utf-8")
        cajas.update(re.findall(r"<name>([^<]*)</name>", t))
        carpetas.add(re.search(r"<folder>([^<]*)</folder>", t).group(1))
    obtenido = {
        "imagenes_jpg": len(ids_jpg),
        "mascaras_png": len(ids_png),
        "xml_cajas": len(ids_xml),
        "imagenes_sin_mascara": len(ids_jpg - ids_png),
        "mascaras_sin_xml": sorted(ids_png - ids_xml),
        "cajas_totales": sum(cajas.values()),
        "carpetas_de_anotador": len(carpetas),
        "huella_jpg": huella(jpg),
        "huella_xml": huella_xml(xml),
        "cajas_por_clase": dict(cajas),
    }
    fallos = 0
    for k, esp in ESPERADO.items():
        ok = obtenido[k] == esp
        fallos += not ok
        print(f"{'OK ' if ok else 'DIF'} {k}: {obtenido[k] if not ok or k.startswith('huella') is False else obtenido[k][:16] + '...'}"
              + ("" if ok else f"  (esperado {esp})"))
    print("Coincide con la Data Card." if not fallos else f"{fallos} diferencias con la Data Card.")
    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main())
