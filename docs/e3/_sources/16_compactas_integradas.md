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

(sec-compactas-integradas)=
# Compactas seleccionadas e inferencia

FCN, U-Net y SegNet usan la arquitectura `a0498`: anchos (64, 64, 64), dropout
en el bloque intermedio, según la regla de selección fijada en la búsqueda E2.
Se recuperaron cinco semillas por familia y se verificaron los 15 checkpoints contra los hashes originales.
Para cada artefacto se eligió la semilla con Dice de validación mediano, sin seleccionar por test.

La siguiente tabla se genera mediante `experiments/integrar_compactas_e3.py` desde
`artifacts/<modelo>/metadata.json`, los mismos archivos que habilitan la inferencia del dashboard.

```{code-cell} python3
from pathlib import Path
import csv
from html import escape
from IPython.display import HTML, display
with Path('../data/compactas_a0498_exportacion.csv').open(encoding='utf-8') as f:
    filas = list(csv.DictReader(f))
columnas = ['modelo', 'semilla', 'estado', 'dice_val_artefacto',
            'error_prob_max', 'torchscript_respaldo_aprueba']
cabecera = ''.join('<th>' + escape(c) + '</th>' for c in columnas)
cuerpo = ''.join('<tr>' + ''.join('<td>' + escape(r[c]) + '</td>' for c in columnas)
                 + '</tr>' for r in filas)
display(HTML('<table><thead><tr>' + cabecera + '</tr></thead><tbody>' + cuerpo + '</tbody></table>'))
```

La media de cinco semillas describe la configuración; el Dice del artefacto corresponde
a una sola semilla. No son el mismo estimador. Las imágenes usadas son las 218 de validación,
agrupadas en 122 grupos para Dice. La exportación se verificó solo en validación; no se reentrenó ninguna red.
La demostración de test añadida después se describe al final de este capítulo.

La prueba de ida y vuelta exige error absoluto máximo de probabilidad inferior a 1e-4
y coincidencia de todas las máscaras a umbral 0,5. Se conserva también la diferencia de Dice.
SegNet expresa el unpool 2x2 de pasos no solapados mediante scatter; las salidas se contrastaron
con el unpool original y los pesos no cambiaron. Esta equivalencia se limita a ese patrón de pooling.

El dashboard carga un artefacto aprobado por vez, verifica su firma y muestra la máscara sobre la imagen recibida.
Una exportación que no supera la tolerancia se conserva en el reporte como fallida y no habilita inferencia.
FCN y U-Net superaron la verificación ONNX. SegNet conserva máscaras idénticas, pero su error
máximo de probabilidad fue 0,0212132, superior a la tolerancia, por lo que permanece deshabilitada.
Se verificó además un respaldo TorchScript de SegNet sobre las 218 imágenes: error máximo cero,
máscaras idénticas y diferencia de Dice cero. Este respaldo requiere PyTorch y no se integra
en el dashboard ONNX ni constituye una prueba de despliegue en Render.
La probabilidad no está calibrada y no se presenta como una garantía de seguridad portuaria.
Los resultados históricos de test del E2 permanecen identificados como históricos: no se atribuyen
a las nuevas arquitecturas a0498. Las diez semillas del E3, el original y la evaluación ampliada siguen pendientes.

Reproducción: ejecutar el exportador con `--compactas`, `--modelos fcn unet segnet` y las rutas
del E2, datos y pesos; después ejecutar `experiments/integrar_compactas_e3.py`.
El cuaderno `training/colab/Exportar_ONNX_E3_a0498_v2.ipynb` incluye la integración para Colab.

## Comparación visual sobre una imagen aleatoria de test

Por solicitud del usuario, el tablero incorpora una demostración sobre el test existente del E2.
El botón selecciona uniformemente una imagen del manifiesto de test; al pedir otra excluye solo
la imagen inmediatamente anterior. No se filtra por dificultad ni por el desempeño de las redes.
La fotografía y la máscara real se muestran arriba; la referencia aparece a la izquierda y los
otros modelos a la derecha. La referencia inicial es TransUNet, líder por media de validación del
benchmark histórico E2, no un ganador declarado de la campaña E3 aún en ejecución.

Las máscaras se precalculan con checkpoints elegidos por validación. Se registra modelo, semilla,
arquitectura y SHA-256 de pesos, imágenes y máscaras. Los Dice e IoU visibles son por imagen a
256 × 256, excluyendo el relleno; no son medias de semillas ni pruebas de superioridad.
Las compactas a0498 se identifican por separado de las demás redes históricas. Un modelo sin
predicción verificada muestra su estado pendiente, nunca una máscara de otro modelo.

Esta demostración supone inspección del test y se registra como exploración visual, no como
una nueva evaluación confirmatoria. No se usan sus resultados para cambiar recetas o seleccionar
semillas. El cierre metodológico deberá declarar esta exposición al test y la evaluación ampliada pendiente.

El paquete de fotografías y máscaras se conecta localmente con `E3_TEST_E2` y `E3_TEST_RAIZ`;
`ABRIR_DASHBOARD.cmd` configura las rutas de este PC. Las imágenes no se incorporan al repositorio
ni se publican automáticamente. Las predicciones precalculadas se guardan en `artifacts/demo_test/`.

### Inventario verificado de la demostración

La tabla siguiente procede de la comprobación SHA-256 de las máscaras del comparador. Su cobertura
describe la demostración, no la campaña estadística del E3.

```{code-cell} python3
:tags: [hide-input]
import json
import pandas as pd
inventario = json.loads(Path('../data/demo_test_verificacion.json').read_text(encoding='utf-8'))
print('Fecha:', inventario['fecha'], '| Máscaras verificadas:', inventario['total_mascaras'])
display(pd.DataFrame(inventario['modelos']).rename(columns={
    'modelo': 'Modelo', 'semilla': 'Semilla', 'arquitectura': 'Arquitectura', 'casos': 'Casos de test'
}))
```
