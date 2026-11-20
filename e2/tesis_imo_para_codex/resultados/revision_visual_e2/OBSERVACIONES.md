# Revisión cualitativa de validación y cierre de test

Se seleccionaron seis casos de grupos distintos a partir de métricas de validación guardadas: dos difíciles, menor área positiva, mayor desacuerdo entre redes, próximo a mediana y alto acuerdo. Criterios y promedios de las diez redes/cinco semillas en `casos.json`. La muestra es dirigida, no representativa de frecuencias de error.

`panel_validacion.png` muestra imagen, máscara y errores de FCN/U-Net/TransUNet con semilla 11 fijada, inferidos localmente en CPU. Verde: coincidencia; rojo: falso positivo; azul: región anotada omitida. Los Dice impresos son de esas inferencias ilustrativas, no promedios de cinco semillas. Los originales y checkpoints no se modificaron.

## Observaciones de los paneles

- `image_0419`: etiqueta inclinada, escena oscura y parcialmente rodeada por una abertura. FCN y U-Net omiten casi toda la región; TransUNet sólo recupera una parte. El liderazgo medio no elimina fallos severos compartidos.
- `image_1538`: escena oscura con varios elementos gráficos. Las tres predicciones son incompletas y presentan regiones falsas fuera de la etiqueta anotada.
- `image_0693`: la etiqueta de menor fracción positiva de validación sí es localizada. En la imagen se aprecia otro rótulo cuya correspondencia con la máscara debe revisarse; no se corrigió la anotación. El tamaño pequeño por sí solo no explica todos los fallos.
- `image_0251`: en esta semilla, TransUNet recupera bien el rombo mientras FCN y U-Net omiten parte de su superficie y confunden elementos vecinos. Es un caso de desacuerdo seleccionado expresamente, no prueba independiente de superioridad.
- `image_1081` y `image_0030`: las tres redes capturan gran parte de las regiones anotadas; predominan pequeñas diferencias de contorno.
- **No hay máscaras vacías en las 218 imágenes de validación.** Por ello, estos datos no permiten estimar tasa de falsas alarmas en fotografías sin ninguna etiqueta. Sí permiten observar falsos positivos sobre fondo dentro de imágenes positivas.

## Decisión

Mantener cerradas las diez recetas y las cinco semillas. No cambiar máscaras, umbral ni hiperparámetros como respuesta a estos ejemplos. Anotar posibles problemas de referencia como limitación y como trabajo futuro.

El usuario autorizó continuar con el test. Se preparó `config/protocolo_test_e2_cerrado.json` con 50 checkpoints identificados por hash, orden de test, umbral 0,5 y procesamiento original. Principal: Dice agrupado. Contraste principal propuesto y fijado antes de test: TransUNet frente a FCN. IC95 por remuestreo de grupos con cinco semillas fijas; dispersión entre semillas por separado. Comparaciones restantes exploratorias, sin afirmar significancia simultánea.

El test de 218 imágenes/122 grupos requiere ejecución GPU de las diez redes, incluido VM-UNet. Paquete separado: `paquete_test_e2_cerrado.zip` y `Evaluacion_Final_E2_Colab.ipynb` en la carpeta superior. No se ejecutó aquí inferencia de test. El test ya tuvo exposición en E1; no debe describirse como un conjunto nunca observado.
