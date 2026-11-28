"""Configuracion de pytest: pone e3/ en sys.path para importar `src` sin instalar el paquete."""
import sys
from pathlib import Path

E3 = Path(__file__).resolve().parents[1]
if str(E3) not in sys.path:
    sys.path.insert(0, str(E3))
