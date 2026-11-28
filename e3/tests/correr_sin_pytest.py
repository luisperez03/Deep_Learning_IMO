"""Corredor minimo para entornos sin pytest (env_e2 no lo trae). Ejecuta cada funcion test_* de
test_original_model.py e informa PASO / FALLO / SALTADA. Con pytest instalado, usar `pytest e3/tests` en su lugar.

Uso: python correr_sin_pytest.py [filtro]   (filtro opcional: subcadena del nombre de la prueba)"""
import importlib
import sys
import time
import traceback
import types
from pathlib import Path

E3 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(E3))
sys.path.insert(0, str(E3 / "tests"))


class _Salto(Exception):
    pass


def _importorskip(nombre, *a, **k):
    try:
        return importlib.import_module(nombre)
    except ImportError:
        raise _Salto(f"falta el modulo {nombre}")


class _Raises:
    def __init__(self, exc, match=None):
        self.exc, self.match = exc, match

    def __enter__(self):
        return self

    def __exit__(self, tipo, valor, tb):
        if tipo is None:
            raise AssertionError(f"no se lanzo {self.exc}")
        if not issubclass(tipo, self.exc):
            return False
        if self.match and self.match not in str(valor):
            raise AssertionError(f"el mensaje '{valor}' no contiene '{self.match}'")
        return True


try:
    import pytest  # noqa: F401
except ImportError:  # un pytest minimo con lo que usan las pruebas
    falso = types.ModuleType("pytest")
    falso.importorskip = _importorskip
    falso.skip = lambda msg="": (_ for _ in ()).throw(_Salto(msg))
    falso.raises = _Raises
    sys.modules["pytest"] = falso

if __name__ == "__main__":
    filtro = sys.argv[1] if len(sys.argv) > 1 else ""
    modulo = importlib.import_module("test_original_model")
    resumen = {"PASO": 0, "FALLO": 0, "SALTADA": 0}
    for nombre in sorted(n for n in dir(modulo) if n.startswith("test_") and filtro in n):
        t0 = time.perf_counter()
        try:
            getattr(modulo, nombre)()
            estado, nota = "PASO", ""
        except _Salto as e:
            estado, nota = "SALTADA", str(e)
        except BaseException as e:
            if type(e).__name__ == "Skipped":
                estado, nota = "SALTADA", str(e)
            else:
                estado, nota = "FALLO", traceback.format_exc()
        resumen[estado] += 1
        print(f"[{estado}] {nombre} ({time.perf_counter() - t0:.1f} s) {nota}", flush=True)
    print(resumen)
    sys.exit(1 if resumen["FALLO"] else 0)
