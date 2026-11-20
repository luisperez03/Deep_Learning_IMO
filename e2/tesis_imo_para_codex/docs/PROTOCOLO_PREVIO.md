# Protocolo previo — propuesta para validar antes de ejecutar

Configuración completa: `config/experimento_propuesto.json`. Fecha: 2026-11-06.
Estado: preparado, no aprobado como protocolo definitivo, sin modelos ejecutados.

## Pregunta y unidad de análisis

Comparar tres familias de segmentación binaria en la localización de rótulos.
Una imagen es la unidad de predicción; las imágenes relacionadas se agrupan.
Métrica primaria propuesta: media de Dice de etiqueta por grupo (promediar primero
las imágenes dentro de cada grupo). Esto limita el peso excesivo de copias.
Reportar también Dice por imagen, IoU de etiqueta, mIoU etiqueta/fondo, exactitud,
tiempo y parámetros. Las unidades de agrupamiento son aproximaciones conservadoras,
no escenas de adquisición certificadas. Declararlo como amenaza de validez.

Píxeles de padding excluidos de métricas y pérdida. Umbral fijo 0,5. Si predicción
y objetivo están vacíos, Dice/IoU de etiqueta valen 1; si solo uno está vacío,
valen 0. Los tests cubren estas convenciones. No escoger umbrales con test.

## Datos y depuración

Se mantienen juntos TODOS los enlaces de similitud conocidos, sin afirmar que
sean duplicados verdaderos. Se propaga cuarentena dentro del grupo cuando hay
alertas de anotación. Clases discrepantes se ponen en cuarentena por precaución,
aunque algunas siluetas puedan servir para binario. No se reasigna automáticamente
la clase de una máscara a partir de una caja.

Las particiones se asignan una vez con semilla 20261106 y cuotas 70/15/15 por
imagen, respetando grupos completos. No se buscan semillas favorables ni se
ajusta a métricas de prueba. El EDA previo exploró todo el corpus: debe declararse
que la futura prueba es una partición interna reservada, no un dataset externo
completamente desconocido. No se mide rendimiento en ella antes del protocolo final.

## Modelos y ajustes

Familias candidatas: FCN pequeña, SegNet pequeña y U-Net pequeña. Diferencias
previstas: reconstrucción desde características gruesas, uso de índices de pooling,
y conexiones de salto de características. Son candidatos, no selección bibliográfica
final. No se fijan arbitrariamente profundidad y ancho atribuyéndolos a artículos.
La literatura o el espacio de búsqueda requerido por la clase deben cerrar esos valores.

Propuesta común: BCE + Dice (pesos 0,5/0,5), Adam, learning rates 0,001 y 0,0003,
mismo presupuesto de dos configuraciones por modelo, máximo 30 épocas y early
stopping de cinco épocas en Dice por grupo de validación. Guardar y recuperar el
mejor checkpoint. Estos números son propuestas de diseño, no resultados optimizados.
Sin pesos preentrenados, sin aumentos al comienzo; si se introducen, hacerlo igual
para todos y registrar la decisión antes de comparar.

## Piloto y estadística

Después de aprobar hardware y arquitectura, prueba mínima de carga y forma con
8 imágenes y hasta 5 pasos. Esa prueba no es el piloto estadístico.

Piloto: cinco semillas comunes por modelo sobre validación. Registrar media, SD,
CV y diferencias pareadas. Delta propuesto = 0,02 puntos absolutos de Dice (dos
puntos porcentuales): requiere justificar relevancia práctica con el investigador;
no es un umbral normativo ni un valor estimado mirando test.

Calcular R aproximadamente como (z_(1-alpha/2)+z_(0.80))² * SD_d² / delta².
Para planificación conservadora de tres contrastes usar alpha/3; para reporte
aplicar Holm. Número definitivo propuesto >=8 y >=R, sujeto a presupuesto validado.
Si no es viable, reportar diferencia mínima detectable y limitación, nunca simular
potencia suficiente. Cinco semillas no permiten significancia bilateral Wilcoxon
a 0,05. No tratar corridas con el mismo test como observaciones nuevas independientes.

Para prueba final: reportar intervalos por remuestreo de grupos de imágenes;
mantener la estructura pareada entre modelos. Separar incertidumbre por semilla
de incertidumbre por muestreo de imágenes. No apilar imagen×semilla como réplicas
independientes. Registrar un plan final de inferencia antes de entrenar.

Test: una evaluación por corrida con configuración fijada. Describir errores
cualitativos después, sin volver a seleccionar hiperparámetros con esa información.
No interpretar ausencia de significancia como equivalencia.

## Reproducibilidad y punto de parada

Registrar hashes de fuentes, splits y máscaras, versiones, semillas, configuraciones,
hardware, hilos, tiempos, curvas y checkpoints. El entorno del EDA está fijado;
el entorno de entrenamiento CPU todavía no se instala ni se declara validado.

Todos los indicadores de autorización permanecen falsos. No hay un comando que
entrene en el flujo entregado. La próxima decisión es validar este protocolo y
el equipo con el investigador, no lanzar automáticamente 15 o más corridas.
