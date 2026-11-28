"""Verifica que los documentos congelados del E3 coinciden con los de la etiqueta e3-protocolo-v2.

  python e3/experiments/verificar_congelamiento.py

La etiqueta se puso el 2026-11-24 a las 20:56 (-05:00) en el repositorio de trabajo privado donde se escribió el
protocolo (commit 4dc8b16033b73ff4263c58cfc56e96230b3876ea), antes de cualquier corrida del E3: las corridas que se
reportan salen de la secuencia v4, fechada el 2026-11-27 (continuidad/SECUENCIA_E3_LISTA.txt). Este repositorio público recibió los documentos después, así que la prueba de que
no cambiaron es el hash de su contenido. Los hashes se calculan con finales de línea normalizados a LF (la copia
pública se guardó con CRLF). En estandar.md se descuenta la nota de auditoría agregada encima del texto el 27/11
(de la línea que empieza con "> **Estado de auditoría local" hasta la primera línea vacía, inclusive); el resto debe coincidir byte a byte.
Sale con código 1 si algún documento no coincide.
"""
import hashlib
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
ETIQUETA = {
    "protocolo.md": "c3d7d85cfdaaa8d82dbe6720c27d13b412ad2291963d13f5b2ef166033cbf24e",
    "estandar.md": "31e1b6634e8a5eb92a04cd83c8c64d5f243f2c67fba204923c4e8f30160996b3",
    "procedencia.md": "b7714c907aa4b8752fc4ba1388aac0d63afc60cec58e9637a9ed4d94b7a71971",
    "design_log.md": "de3553bae1be27fe2c7d1ce671332b40dab92ce4bc7f42f6d43309f70a2bfb49",
}


def contenido(nombre: str) -> bytes:
    texto = (AQUI / nombre).read_bytes().replace(b"\r\n", b"\n")
    if nombre == "estandar.md":
        lineas = texto.split(b"\n")
        ini = next((i for i, l in enumerate(lineas) if l.startswith(b"> **Estado de auditor")), None)
        if ini is not None:
            fin = lineas.index(b"", ini)  # la nota termina en la primera línea vacía, que también se quita
            lineas = lineas[:ini] + lineas[fin + 1:]
        texto = b"\n".join(lineas)
    return texto


def main() -> int:
    fallos = 0
    for nombre, esperado in ETIQUETA.items():
        h = hashlib.sha256(contenido(nombre)).hexdigest()
        ok = h == esperado
        fallos += not ok
        print(("IGUAL    " if ok else "DISTINTO "), nombre, h)
    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main())
