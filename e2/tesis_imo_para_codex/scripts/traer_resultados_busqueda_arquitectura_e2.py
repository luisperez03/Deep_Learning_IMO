"""Copia el paquete liviano de la búsqueda de arquitectura (sin pesos) a resultados/busqueda_arquitectura_e2/.

  python scripts/traer_resultados_busqueda_arquitectura_e2.py RUTA/busqueda_arquitectura_e2_resultados.zip

El zip lo deja la celda «6. Resumen» de colab/Busqueda_Arquitectura_E2.ipynb en la carpeta de Drive
tesis_imo_e2_busqueda_arquitectura_v1 (se descarga a mano). Rechaza miembros con ruta absoluta o con «..»
y cualquier .pt; no borra los archivos previos de la carpeta (prueba_cpu.json y las estimaciones locales).
"""
import sys
import zipfile
from pathlib import Path, PurePosixPath

RAIZ = Path(__file__).resolve().parents[1]
DESTINO = RAIZ / "resultados" / "busqueda_arquitectura_e2"


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    zpath = Path(sys.argv[1])
    with zipfile.ZipFile(zpath) as z:
        nombres = z.namelist()
        for n in nombres:
            p = PurePosixPath(n)
            if p.is_absolute() or ".." in p.parts or p.suffix == ".pt":
                raise SystemExit(f"Miembro no permitido: {n}")
        z.extractall(DESTINO)
    fase2 = sum(1 for n in nombres if n.startswith("fase2/") and n.endswith("result.json"))
    fase1 = sum(1 for n in nombres if n.startswith("fase1/") and n.endswith("result.json"))
    print(f"{len(nombres)} archivos en {DESTINO}; result.json fase 1 = {fase1}, fase 2 = {fase2}")


if __name__ == "__main__":
    main()
