---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python libro E3
  language: python
  name: python3
---

(sec-avance-colab)=
# Resultados recibidos de Colab

El paquete `tesis_imo_e3_secuencia_v4-20261128T013803Z-1-001.zip` contiene veinte corridas
completas del original y nueve de SegFormer corregido. Se verificaron configuración,
identidad, huella de cada mejor checkpoint, número de épocas y coincidencia entre la mejor
validación del historial y `result.json`. El nombre del archivo usa fecha UTC; este registro
se integra en la jornada local del 27 de noviembre.

## Resultados de validación

```{code-cell} python3
:tags: [hide-input]
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
root=Path('../data')
report=json.loads((root/'avance_colab.json').read_text(encoding='utf-8'))
summary=pd.DataFrame(report['resumen'])
display(summary[['modelo','n','esperadas','media','de','minimo','maximo','horas','estado']].round(4))
print('Diferencia media pareada UNet-AT − base:',round(report['diferencia_val_at_menos_base'],4))
```

Cada valor corresponde al mejor Dice por grupo en validación de una corrida, utilizado
para seleccionar época. La dispersión es entre semillas; no es un intervalo de confianza.
SegFormer aún tiene una semilla pendiente y su media puede cambiar. No se combinan estos
valores con la tabla histórica de test del E2.

## Comparación de los dos brazos

```{code-cell} python3
:tags: [hide-input]
runs=pd.read_csv(root/'avance_colab_corridas.csv')
original=runs[runs.modelo.str.startswith('unetat_')]
paired=original.pivot(index='semilla',columns='modelo',values='dice_val').sort_index()
fig,ax=plt.subplots(figsize=(7,4))
for seed,row in paired.iterrows():
    ax.plot([0,1],[row['unetat_dbnet'],row['unetat_dbnet_at']],marker='o',alpha=.65,label=str(seed))
ax.set_xticks([0,1],['Base DBNet','UNet-AT'])
ax.set_ylabel('Mejor Dice de validación por grupo')
ax.set_title('Comparación pareada por semilla; selección de época en validación')
ax.legend(title='Semilla',bbox_to_anchor=(1.02,1),loc='upper left',fontsize=8)
fig.tight_layout();plt.show()
```

UNet-AT obtiene media 0,7862 frente a 0,8077 de su base: diferencia −0,0215.
El resultado no respalda una mejora global en esta validación. No se declara todavía
refutada o confirmada la hipótesis específica del tercil pequeño, porque requiere
evaluación por tamaño y la comparación final definida en el protocolo. Tampoco se
atribuye la diferencia a C1 o C2 sin sus ablaciones.

## Curvas de entrenamiento y validación

```{code-cell} python3
:tags: [hide-input]
curves=pd.read_csv(root/'avance_colab_curvas.csv')
fig,axes=plt.subplots(2,3,figsize=(12,6),sharex=True)
for j,name in enumerate(['unetat_dbnet','unetat_dbnet_at','segformer_b0']):
    for seed,part in curves[curves.modelo==name].groupby('semilla'):
        axes[0,j].plot(part.epoca,part.perdida,alpha=.5)
        axes[1,j].plot(part.epoca,part.dice_val,alpha=.5)
    axes[0,j].set_title(name)
    axes[1,j].set_xlabel('Época')
axes[0,0].set_ylabel('Pérdida de entrenamiento')
axes[1,0].set_ylabel('Dice de validación por grupo')
fig.tight_layout();plt.show()
```

Las pérdidas del original y de SegFormer tienen definiciones distintas; su escala no
permite compararlos directamente. Las curvas terminan según el límite de épocas o la
parada temprana de cada corrida. Las horas de la tabla suman los tiempos por época
registrados, incluyendo validación; no equivalen a unidades facturadas por Colab.

## Incorporación al tablero y cierre pendiente

El tablero lee `data/avance_colab.json`, generado por `experiments/integrar_avance_colab.py`.
El artefacto representativo del original se selecciona por la mediana inferior de validación
entre las diez semillas (semilla 11). La demostración aplica el posproceso C2 completo;
la salida de probabilidad del núcleo por sí sola no es la máscara final del rótulo.

Faltan la última semilla de SegFormer, las nuevas corridas de los demás rivales y la revisión
previa a la evaluación final. La inspección visual de test se mantiene separada del cierre
estadístico. Los capítulos previos sobre limitaciones y tareas abiertas siguen aplicándose.
