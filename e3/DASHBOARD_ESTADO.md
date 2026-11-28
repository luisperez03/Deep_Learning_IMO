# Dashboard del Entregable 3: actualización (2026-11-27)

FCN y U-Net a0498 tienen inferencia real ONNX local, superposición de máscara y comprobación SHA-256.
Ambas superaron la comparación con PyTorch sobre 218 imágenes de validación. SegNet ONNX no superó
la tolerancia y permanece deshabilitada; su respaldo TorchScript fue verificado con error cero en esas
218 imágenes, pero no está integrado en la aplicación. No se evaluó test ni se gastó GPU.
Las 24 pruebas de dashboard, consistencia e integración pasaron. No se publicó en Render.
Ver `../LEER_PRIMERO.md` para abrir la aplicación y `artifacts/` para los metadatos.

## Registro histórico (2026-11-23)

> Trazabilidad. Esqueleto funcional generado el 2026-11-23 a partir de los resultados del E2 (solo lectura). Verificado con
> `pytest tests/test_dashboard.py` (11 pruebas) y con el servidor local (`/`, `/eda`, `/modelos` devuelven 200, 9 callbacks
> registrados, un callback del EDA invocado por HTTP). No desplegado en Render.

## Qué hay

- `app.py`: Dash multipágina (`use_pages`), `server = app.server`, tema FLATLY, navbar fija, cabeceras básicas, pie con privacidad.
- `pages/contexto.py`, `pages/eda.py`, `pages/modelos.py`; `src/data.py` (lectura), `src/plots.py` (plantilla única, paleta de
  5 colores, símbolos y patrones además del color), `src/inference.py` (stub), `src/ui.py`, `assets/style.css`.
- `requirements.txt` (versiones exactas, sin torch) y `render.yaml` (gunicorn 1 worker, 2 hilos, Python 3.11.9).

## Datos y su script (fuente: `tesis_imo_para_codex` del E2, vía `E3_E2_RESULTADOS`)

| Archivo en `data/` | Script | Contenido |
|---|---|---|
| `tabla_maestra.csv` (+ `_meta.json` con SHA-256 de las entradas) | `experiments/generar_tabla_maestra.py` | 10 redes + baseline + fila original "pendiente". Dice recalculado desde `por_imagen_corridas.csv` y verificado contra el E2 (falla si no reproduce); DE e IC95 del informe del E2 (el IC recalculado coincide a ±0,003) |
| `eda_imagenes.csv`, `dice_por_caso.csv`, `dice_por_area.csv`, `limitacion_area.csv`, `calidad_shift.csv`, `calidad_resumen.json` | `experiments/generar_tablas_dashboard.py` | Derivadas livianas; sin imágenes ni máscaras |

Regenerar: definir `E3_E2_RESULTADOS` (ruta a `tesis_imo_para_codex`) y correr ambos scripts.

## Marcadores y por qué

- Modelo original (fila, hipótesis, subsección): pendiente hasta congelar el protocolo (bloques C y D).
- Calibración (ECE): pendiente en todas las filas; no se midió en el E2.
- Predicción en vivo: valida formato y tamaño (<= 5 MB), avisa fuera de dominio (reglas R1 a R4, **umbrales sin calibrar**,
  documentados en `src/inference.py`) y responde "Artefacto ONNX pendiente". Falta el bloque B (exportación ONNX);
  `preprocesar` está probado, la ejecución de la sesión es un TODO.
- Ejemplos de máscaras: pendiente (se generan al exportar).
- Autores del pie: por completar. Enlace a la model card: apunta a una ruta aún inexistente (`E3_MODEL_CARD_URL` lo sobrescribe).
- IoU: TP/(TP+FP+FN) por imagen; mIoU y exactitud no se incluyen. VM-UNet no tiene latencia en CPU (requiere CUDA).

## Riesgos

1. Licencia de las imágenes de HAZMAT-13 sin confirmar: no se copia ninguna; la galería exige `E3_DATOS_IMAGENES` y en Render
   mostrará solo el aviso.
2. Cifra de relevancia (Tianjin 2015: 165 fallecidos, 8 desaparecidos, 798 heridos; informe del Consejo de Estado, feb. 2016,
   vía China Daily) tomada de una búsqueda web resumida: revisar contra la fuente primaria antes de entregar.
3. Cuarentena (132), sin máscara (102) y pares similares (748/146) salen de prosa del E2; están como constantes con su fuente en
   `generar_tablas_dashboard.py`.
4. Sin Content-Security-Policy (Dash usa scripts en línea); solo `nosniff`, `X-Frame-Options` y `Referrer-Policy`.
5. Algunos textos de interpretación de la página Modelos son estáticos y describen las cifras actuales del E2.

## Cómo correrlo

    cd e3
    python app.py                                  (http://127.0.0.1:8050/)
    pytest tests/test_dashboard.py
    gunicorn app:server --workers 1 --threads 2 --timeout 120   (en Render: pip install -r requirements.txt)
