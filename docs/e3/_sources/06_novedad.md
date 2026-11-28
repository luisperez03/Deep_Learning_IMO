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

(sec-novedad)=
# Revisión de novedad

Mecanismo evaluado: encogimiento del núcleo atenuado por el radio inscrito (C1) y expansión consistente en inferencia
como punto fijo con expansión en inglete (C2), sobre la representación de DBNet {cite:p}`liao2020dbnet`, aplicado a la
segmentación binaria de rótulos de mercancías peligrosas.

## Cadenas de búsqueda y matriz

Ocho cadenas (C1 a C8) combinan al menos dos de tres ejes: el mecanismo, el problema de aplicación y los sinónimos en
inglés y español. El texto de cada cadena, la lógica por fuente y el código que las envió están en
`experiments/novedad/cadenas.md` y `experiments/novedad/buscar.py`; las respuestas crudas, en
`experiments/novedad/resultados/` (OpenAlex, arXiv, Semantic Scholar, OpenReview, GitHub y Hugging Face Hub).

```{code-cell} python3
:tags: [hide-input]
# Matriz fuente x cadena: número de archivos de respuesta guardados por cada combinación.
from pathlib import Path
import pandas as pd

res = Path("..") / "experiments" / "novedad" / "resultados"
filas = []
for f in sorted(res.glob("*__*.json")):
    fuente, cadena = f.stem.split("__", 1)
    filas.append({"fuente": fuente, "cadena": cadena})
m = pd.DataFrame(filas)
display(pd.crosstab(m["fuente"], m["cadena"]))
```

## Declaración y verificación por un tercero

```{admonition} Pendiente
:class: warning
- **Declaración de novedad:** redactar con lo que arrojen el cribado (`cribar.py`) y el rastreo de citas
  (`snowball.py`); hoy no hay conclusión escrita.
- **Bases de suscripción:** Scopus, Web of Science e IEEE Xplore (sintaxis en `pendiente_bases_suscripcion.md`, si
  existe en la carpeta `novedad/`).
- **Verificación por un tercero:** un compañero ajeno al diseño repite al menos dos cadenas; guía en
  `experiments/novedad/guia_verificador.md`. Se reportará qué cadenas repitió, qué encontró de más o de menos y el
  acuerdo con la matriz.
```
