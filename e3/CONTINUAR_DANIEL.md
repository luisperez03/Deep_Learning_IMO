# Continuidad del E3 — corte del 27 de noviembre de 2026

## Qué está listo

- Libro fuente en `book/`; versión HTML en `../docs/e3/` y ZIP descargable del release.
- Tablero Dash: contexto, EDA, benchmark E2, avance Colab y comparador aleatorio de test.
- Original: 10 semillas `dbnet` y 10 `dbnet_at`, verificadas por SHA-256 y consistencia de historial.
- SegFormer corregido: 9 semillas; falta **107**. No mezclar con la receta antigua.
- Auditoría completa: **74/120** corridas verificadas contando los resultados E2 reutilizables.
- Comparador: diez redes con predicciones precalculadas, incluido UNet-AT semilla 11 y C2 completo. VM-UNet pendiente.
- FCN y U-Net a0498 admiten fotografías propias mediante ONNX. La demostración precalculada del original no equivale a inferencia ONNX en vivo.

## Resultado que debe conservarse en el informe

Mejor Dice de validación por corrida (media ± DE entre semillas): base 0,8077 ± 0,0109;
UNet-AT 0,7862 ± 0,0192; SegFormer parcial 0,8090 ± 0,0143.
UNet-AT − base = −0,0215. No hay evidencia de mejora global en esta validación.
La hipótesis por tamaño, las ablaciones y la comparación final de test siguen pendientes.
No cambiar recetas para mejorar estos resultados observados ni presentar la demostración como prueba confirmatoria.

## Descargas y datos

Release de avance: `https://github.com/luisperez03/Deep_Learning_IMO/releases/tag/e3-avance-20261127`.

1. `tesis_imo_e3_secuencia_v4-20261128T013803Z-1-001.zip`: paquete recibido, con checkpoints y estados para continuar. Extraerlo en una carpeta propia; no sobrescribir una campaña más reciente.
2. `artefactos_dashboard_e3.zip`: extraer desde la raíz del repositorio; crea `e3/artifacts/` con pesos de inferencia y máscaras verificadas.
3. `libro_e3_html.zip`: extraer y abrir `intro.html` para leer el libro sin servidor Dash.
4. `original_e3_20_corridas.zip`: descarga alternativa de 34 MB con solo el original. Es un subconjunto del paquete completo; no hace falta descargar ambos para trabajar con ese modelo.

Las huellas de los paquetes están en `continuidad/*.sha256`. Los registros ligeros recibidos se conservan en
`continuidad/registros_colab/`; contienen rutas del entorno de origen que deben adaptarse al equipo receptor.
Las fotografías y máscaras reales del dataset **no se redistribuyen** en GitHub.
Se necesita el paquete E2 de datos y máscaras que ya usa Colab (`paquete_colab_e2_v2.zip`), las compactas
recuperadas (`compactas_a0498_e3.zip`) y los checkpoints históricos para auditar las 120 corridas.
Si falta alguno, solicitarlo a Luisa; no inventar rutas ni regenerar particiones.

## Levantar el tablero

Crear un entorno Python, instalar `e3/requirements.txt` y descargar los artefactos del release.
Para el comparador, definir `E3_TEST_E2` como la carpeta que contiene `resultados/preentrenamiento/test.csv`
y las máscaras binarias; `E3_TEST_RAIZ` como la carpeta que contiene las rutas `image_path` del CSV
(habitualmente `VOC2012/JPEGImages`). Las huellas de los archivos deben coincidir.

Ejemplo PowerShell desde la raíz del repositorio, sustituyendo las dos rutas de datos por las reales:

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r e3/requirements.txt
$env:E3_TEST_E2 = 'RUTA_REAL_DEL_E2'
$env:E3_TEST_RAIZ = 'RUTA_REAL_RAIZ_IMAGENES'
$env:HOST = '::1'
$env:PORT = '8051'
.venv/Scripts/python e3/app.py
```

Abrir `http://[::1]:8051/modelos`. En Linux u otros equipos puede utilizarse `HOST=127.0.0.1` y otro puerto.
Sin el dataset, las tablas siguen disponibles; el comparador informa que falta conectar el paquete.

## Continuar Colab sin perder avances

Usar `continuidad/E3_CONTINUAR_EN_ESTA_MAQUINA.py` o el cuaderno único del mismo directorio.
El carril `todo` conserva etapas completas y reanuda las incompletas verificando su identidad.
No ejecutar simultáneamente `todo` y otros carriles sobre las mismas salidas. No cambiar GPU/versiones
en una corrida parcial sin revisar la compatibilidad del checkpoint; no borrar resultados completos.
La carpeta de salida es `MyDrive/tesis_imo_e3_secuencia_v4`.

Faltan: SegFormer 107, 15 corridas compactas, 10 rivales A y 20 rivales B.
La búsqueda del original (04), auditoría (09) y evaluación de validación (10) son etapas adicionales;
las 120 corridas de la auditoría no incluyen los ensayos de búsqueda.

## Incorporar el siguiente paquete

```text
python e3/experiments/integrar_avance_colab.py --campana RUTA_CAMPAÑA --salida e3/data
```

Usar una copia acumulada de la campaña: el integrador reemplaza el resumen con las corridas presentes,
no combina automáticamente paquetes sueltos. Verificar resultados antes de sustituir archivos existentes.
`auditar_campana.py` requiere además los pesos históricos y las compactas; devuelve código 2 si faltan corridas.
Eso significa campaña incompleta, no necesariamente corrupción.

Compilar el libro con Jupyter Book 0.x (no la versión mayor nueva), usando un kernel que disponga de
pandas y matplotlib. `continuidad/entorno_libro_local.freeze.txt` registra el entorno efectivamente probado.
Ejecutar `jupyter-book build e3/book -W --keep-going` y copiar el HTML a `docs/e3/` cuando apruebe.

## Pendientes científicos y de entrega

Evaluación final por imagen/grupo, análisis por tamaño, ablaciones, robustez, potencia actualizada,
revisión de novedad y demostración, tres conjuntos adicionales, política final uniforme y despliegue público.
El original no se declara superior. No se recibió todavía retroalimentación del docente sobre E2.
El libro es una entrega de avance presentable, no una certificación de cumplimiento completo del enunciado.
