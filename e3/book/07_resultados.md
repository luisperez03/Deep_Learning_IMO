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

(sec-resultados)=
# Resultados

```{admonition} Actualización: paquete de Colab recibido
:class: note
Se integraron 20 corridas del original y 9 de SegFormer, verificadas contra sus checkpoints.
Las tablas, curvas y la lectura de estos resultados están en {ref}`sec-avance-colab`.
Las menciones a evaluación pendiente se refieren al cierre final, no a la ausencia de entrenamiento.
```


## Benchmark histórico de las diez redes (E2)

Todas las cifras de esta sección salen de `data/tabla_maestra.csv`, la misma tabla que lee el dashboard
(`src/data.py`); no hay medias copiadas a mano. La tabla se genera con `experiments/generar_tabla_maestra.py` desde los
resultados del E2 y verifica que el Dice recalculado reproduzca el del E2.

```{code-cell} python3
:tags: [hide-input]
# Tabla principal del benchmark: se lee de data/tabla_maestra.csv (versionado) y se ordena por Dice en test.
import hashlib
from pathlib import Path
import pandas as pd

csv = Path("..") / "data" / "tabla_maestra.csv"
print("SHA-256 de tabla_maestra.csv:", hashlib.sha256(csv.read_bytes()).hexdigest())
t = pd.read_csv(csv)
cols = ["nombre", "dice_test", "de_semillas", "ic95_inf", "ic95_sup", "iou_test", "parametros", "gflops", "estado"]
tabla = t[cols].sort_values("dice_test", ascending=False, na_position="last").reset_index(drop=True)
tabla.columns = ["Modelo", "Dice test", "DE entre semillas", "IC95 inf", "IC95 sup", "IoU test",
                 "Parámetros", "GFLOPs", "Estado"]
display(tabla.style.format({"Dice test": "{:.4f}", "DE entre semillas": "{:.4f}", "IC95 inf": "{:.4f}",
                            "IC95 sup": "{:.4f}", "IoU test": "{:.4f}", "Parámetros": "{:,.0f}",
                            "GFLOPs": "{:.2f}"}, na_rep="pendiente").hide(axis="index"))
```

```{code-cell} python3
:tags: [hide-input]
# Figura 1: Dice en test por red, con intervalo de confianza del 95 % por grupos (barras de error).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = t.dropna(subset=["dice_test"]).sort_values("dice_test")
err = [d["dice_test"] - d["ic95_inf"], d["ic95_sup"] - d["dice_test"]]
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.barh(d["nombre"], d["dice_test"], xerr=err, color="#4C78A8", capsize=3)
ax.set_xlim(0.6, 1.0)
ax.set_xlabel("Coeficiente de Dice en test (adimensional, 0 a 1)")
ax.set_ylabel("Red")
ax.set_title("Figura 1. Dice en test por red (media sobre semillas; IC 95 % por grupos)")
fig.tight_layout()
from io import BytesIO
from IPython.display import Image
buffer = BytesIO()
fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
display(Image(data=buffer.getvalue()))
plt.close(fig)
```

*Figura 1.* Dice por grupo en test de cada red. Las filas sin valor (el modelo original) no se dibujan hasta que existan
sus corridas.

## Modelo original frente a los diez rivales

```{admonition} Pendiente
:class: warning
La secuencia vigente es `training/colab/secuencia_v4/`, ejecutable con el lanzador único sin cambiar de máquina. Se preparan 20 corridas del original y 100 de rivales: 45 históricas reutilizables y 55 nuevas. `experiments/auditar_campana.py` comprueba la campaña y `experiments/evaluar_campana.py` prepara la evaluación de validación. El cierre estadístico de test sigue pendiente; no se sustituyen sus resultados por los del E2.
```

La tabla siguiente fija el formato de la comparación; sus celdas se llenan desde ese script.

```{list-table} Pruebas de hipótesis de UNet-AT frente a cada rival (formato; datos pendientes)
:header-rows: 1
:name: tab-pruebas

* - Prueba
  - H0
  - Supuestos verificados
  - Estadístico
  - p crudo
  - p ajustado (Holm, m = 10)
  - Tamaño del efecto
  - IC 95 %
  - Conclusión
* - Wilcoxon de rangos con signo, pareado por grupo
  - Mediana de $d_j$ igual a 0
  - Pares independientes entre grupos; simetría de $d_j$ (gráfico)
  - Pendiente
  - Pendiente
  - Pendiente
  - $\delta$ de Cliff
  - Bootstrap por grupos
  - Pendiente
* - Bootstrap pareado por grupos (10.000 réplicas)
  - Media de $d_j$ igual a 0
  - Grupos intercambiables
  - Pendiente
  - Pendiente
  - Pendiente
  - Diferencia media
  - Percentil
  - Pendiente
* - t pareado por grupo
  - Media de $d_j$ igual a 0
  - Normalidad de $d_j$ (Shapiro-Wilk, reportado)
  - Pendiente
  - Pendiente
  - Pendiente
  - $d$ de Cohen pareado
  - t
  - Pendiente
```

**Condición de superioridad** (protocolo, §5): el mayor de los tres p ajustados es menor que 0,05, la diferencia media
es al menos 0,02, $\lvert\delta\rvert\ge 0{,}147$ y el límite inferior del IC 95 % es mayor que 0 {cite:p}`holm1979`
{cite:p}`wilcoxon1945` {cite:p}`cliff1993`.

## Complementos que no deciden

Friedman con Iman-Davenport, Nemenyi y diagrama de diferencia crítica {cite:p}`demsar2006`, modelo de efectos mixtos y
probabilidad de superioridad: pendientes del mismo script. El valor de la diferencia crítica de Nemenyi del
benchmark del E2 está en `data/tabla_maestra_meta.json`.
