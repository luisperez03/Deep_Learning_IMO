# Piloto local reproducible

Preparación autorizada el 7 de noviembre de 2026. Configuración independiente en
`config/piloto_local.json`; no se cambia la preparación congelada ni se declara
aprobado el benchmark definitivo por ejecutar este piloto.

## Ejecución

Ejecuta `PILOTO_LOCAL.cmd`, cierra ChatGPT por completo y pulsa una tecla en la
consola. Conecta el cargador y conserva la ventilación del portátil. Mantén el
equipo encendido; el script no modifica los ajustes de energía ni impide suspensión.
Puede tardar muchas horas. Para medir únicamente el presupuesto antes de dejarlo
trabajando, usa `MEDIR_PRESUPUESTO.cmd`; sus épocas se aprovechan al continuar.

Primero comprueba numéricamente la reanudación con un modelo sintético mínimo.
Después verifica hashes y mide una época de cada modelo con semilla 11. Guarda
`presupuesto.json` con la proyección sin early stopping y continúa hasta cinco
semillas (11,23,37,53,71) por modelo. La época exploratoria FCN anterior permanece
separada: no se utiliza su checkpoint para alterar el inicio de las corridas.

Condiciones comunes: CPU, dos hilos, lote 1, 256 píxeles, Adam 0.001, BCE/Dice
0.5/0.5 sin padding, máximo 30 épocas y paciencia 5 en Dice de validación medio
por grupo. Entrenamiento desde cero. Una corrida a la vez. Test no se lee.
Una tasa fija para el piloto; no completa la búsqueda de dos tasas propuesta.

## Interrupciones y resultados

Ctrl+C detiene. Repetir el mismo archivo retoma la última época completa; una
época interrumpida se repite. Comprueba la RAM antes de cada imagen, también en
validación, y detiene si baja de 2 GiB. No mide temperatura ni garantiza evitar
picos entre comprobaciones. No se cierran aplicaciones automáticamente.

Cada corrida en `resultados/piloto_local_v1` guarda `last.pt`, `best.pt`, historial,
predicciones de validación del mejor checkpoint, entorno y configuración. Las
semillas de modelos distintos son unidades pareadas para el futuro análisis.
`resumen.json` contiene resultados completos e incompletos, distinguidos por
`completed`. No comparar una corrida incompleta con las terminadas como si tuviera
el mismo presupuesto. `ultima_interrupcion.txt` conserva el último fallo histórico.

La identidad fija código, paquetes instalados, Python, particiones y hashes de
los datos. Un cambio rechaza reanudar silenciosamente. No editar código o versiones
durante el piloto. No ejecutar dos copias a la vez: Windows bloquea la carpeta.
Los checkpoints son archivos privados propios; no cargar los de terceros.

Los seeds y el entorno se registran, pero SegNet tiene advertencia de unpooling
no determinista. El test de reanudación verifica el mecanismo con una convolución
determinista, no demuestra igualdad bit a bit de todas las arquitecturas.

Estado de entrega: sintaxis comprobada; la prueba numérica de reanudación debe
pasar al ejecutar el archivo antes de empezar. No hay quince corridas completadas
ni presupuesto de U-Net/SegNet medido todavía. La preparación local nueva no
actualiza el ZIP de Colab anterior, que permanece como versión separada.
