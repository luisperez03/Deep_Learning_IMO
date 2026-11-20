# Prueba local en Windows

El entorno `.venv-cpu` es independiente del EDA. No usa la Radeon ni descarga
pesos preentrenados. Python base usado: 3.12.14.

Desde esta carpeta, con un Python 3.12 instalado, para reproducir el entorno:

```powershell
python -m venv .venv-cpu
& .venv-cpu/Scripts/python.exe -m pip install -r requirements-cpu.lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

El archivo de versiones se genera al terminar la instalación. Para ejecutar
la prueba autorizada con el entorno ya preparado:

```powershell
& .venv-cpu/Scripts/python.exe scripts/prueba_carga.py --model fcn
```

Solo intenta cinco pasos sobre cinco imágenes de train, lote 1, tamaño 256,
dos hilos y semilla 11. No guarda pesos ni lee test. Si hay menos de 2 GiB
de RAM física libre, se detiene antes de importar PyTorch. Esa reserva es una
regla operativa de este experimento, no una garantía térmica ni una frontera
universal de seguridad. No cerrar aplicaciones ajenas automáticamente.

El resultado se guarda en `resultados/preentrenamiento/carga_fcn.json`:

- `completed_smoke_only`: completó cinco pasos; no significa piloto terminado.
- `blocked_free_ram_below_2_GiB`: no inició por falta de reserva.
- `stopped_low_free_ram`: se detuvo al detectar poca memoria entre pasos.
- `failed`: conservar el error para corregir antes de continuar.

Si termina, los tiempos incluyen carga de imágenes y actualización de pesos.
Cinco pasos no representan una época completa, validación ni entrenamiento
sostenido. No estimar con ellos una duración garantizada de las quince corridas.
La temperatura no se mide en esta prueba.

## Segunda medición autorizada

FCN completó cinco pasos con 3,38 GiB libres inicialmente y 3,08 GiB al terminar
(resultado del 7 de noviembre, 00:38 hora de Colombia). Esto no valida los otros
modelos ni una carga sostenida. El investigador autorizó continuar con U-Net,
SegNet y medir una época completa con validación.

Ejecutar `MEDIR_MODELOS_LOCAL.cmd` con ChatGPT cerrado: prueba U-Net y SegNet
secuencialmente; si ambas terminan, mide una época FCN con las 1015 imágenes de
train y las 218 de val. Cualquier fallo detiene la secuencia. No inicia el piloto.
El reporte está en `resultados/medicion_local/epoca_fcn.json` y registra tiempos
separados, hashes, entorno, memoria muestreada y Dice por grupo. Ese Dice es
exploratorio tras una sola época, no un resultado del benchmark definitivo.
La prueba local usa ahora determinismo con advertencias (igual que Colab),
porque el unpooling de SegNet puede carecer de implementación estrictamente
determinista. Los logs conservan las advertencias; no se afirma igualdad bit a bit.

Verificado: sintaxis y seis pruebas de preparación. La segunda medición todavía
no se ha ejecutado numéricamente: al abrir el chat quedó menos de 1 GiB libre.
