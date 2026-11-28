---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(sec-diseno)=
# Diseño experimental: protocolo y potencia

El protocolo recuperado se incluye tal como está en `experiments/protocolo.md`. La etiqueta
`e3-protocolo` no está verificada localmente. Su inclusión conserva el antecedente, no certifica
que las condiciones de congelamiento se hayan cumplido. La ejecución vigente utiliza la secuencia v4.
La potencia proviene de `experiments/potencia_e3.py` y `experiments/potencia_e3.json`: usa la U-Net
histórica como referencia y no estima todavía la potencia del original ni de las compactas a0498.
Antes del cierre deben reconciliarse recetas, política de entrenamiento final, familia de contrastes
y unidad de análisis; no se reemplaza esa revisión con el número de corridas completadas.

```{include} ../experiments/protocolo.md
:start-line: 10
```

(sec-potencia)=
## Potencia por contraste

```{code-cell} python3
:tags: [hide-input]
# Lee la potencia precalculada (experiments/potencia_e3.json) y la muestra como tabla.
# El script que la genera es experiments/potencia_e3.py; aquí no se recalcula.
import json
from pathlib import Path
import pandas as pd

raiz = Path("..").resolve()
pot = json.loads((raiz / "experiments" / "potencia_e3.json").read_text(encoding="utf-8"))
cols = ["rival", "sigma_d", "R_necesario", "R_necesario_IC80_sup", "MDD_R10", "potencia_R10"]
nombres = ["Rival", "Desv. est. de la diferencia", "R necesario (alfa Holm, potencia 0,90)",
           "R necesario (cota sup. IC 80 %)", "Diferencia mínima detectable con R = 10", "Potencia con R = 10"]
filas = pd.DataFrame(pot["validacion"]["filas"])[cols]
filas.columns = nombres
print("Contrastes en validación con la U-Net compacta como base; delta =", pot["delta"], "; m =", pot["m"])
display(filas.round(4))
```
