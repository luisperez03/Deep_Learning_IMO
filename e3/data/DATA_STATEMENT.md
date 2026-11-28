# Declaración de datos: HAZMAT-13

> **Trazabilidad.** Redactada el 2026-11-23 a partir de `docs/DATA_CARD_HAZMAT13.md` del repositorio del E2
> (`tesis_imo_para_codex`), escrita el 2026-11-22 con las páginas públicas de los autores y la copia local. Este
> directorio no contiene imágenes ni máscaras: solo tablas derivadas.

## Procedencia

| Campo | Valor |
|---|---|
| Nombre | HAZMAT-13, *Hazardous Materials Signs Dataset* |
| Autores | Laboratorio de Robótica Móvil Avanzada (MRL-AMRL), Universidad Azad de Qazvin, Irán, con coautoría de la Universidad de Leeds |
| Repositorio | github.com/mrl-amrl/HAZMAT13 (commit `ec85498`, 12 de julio de 2021); las imágenes están en carpetas de Google Drive enlazadas desde el README |
| Artículo | Sharifi, Zibaei y Rezaei (2021), Machine Learning with Applications, 100104, doi:10.1016/j.mlwa.2021.100104 |
| Versión usada | «Original model»: 1.685 imágenes sin aumento, 13 clases, formato Pascal VOC |
| Uso aquí | Segmentación binaria del rótulo (unión de las máscaras de las 13 clases) |

La copia local coincide con lo publicado en conteo, clases y formato; no se certifica igualdad binaria porque los
autores no publican sumas de verificación.

## Partición

Por grupos de similitud perceptual (semilla 20261106): 1.015 de entrenamiento, 218 de validación y 218 de prueba;
234 imágenes excluidas (102 sin máscara y 132 en cuarentena por discrepancias). La partición y los SHA-256 por imagen
y máscara están en `resultados/preentrenamiento/freeze_manifest.json` del repositorio del E2.

## Licencia

- El repositorio de los autores tiene licencia MIT para «the Software». **No se puede extender con certeza a las
  imágenes ni a las anotaciones**, y la titularidad de las imágenes provenientes de competencias no está documentada.
- **Licencia sin confirmar.** Pendiente: correo a los autores pidiendo aclarar la licencia de los datos, la titularidad
  y las condiciones de redistribución. No se ha enviado.
- Mientras tanto: uso para investigación y docencia con atribución y cita; no se redistribuyen imágenes, máscaras ni
  cajas (el dashboard exige una ruta local para la galería).

## Sesgos y limitaciones conocidos

- Sin EXIF ni campos de origen en los XML: no se puede reconstruir fecha, sensor, lugar ni sesión de cada imagen; la
  agrupación perceptual no sustituye la independencia por escena.
- Geografía no documentada; el test mide generalización interna, no una operación portuaria independiente.
- Sin protocolo de anotación ni acuerdo entre anotadores publicados; el E2 midió un techo por segunda anotación de 50
  imágenes de validación (Dice 0,941).
- Clases desbalanceadas (de 79 cajas de *poison* a 406 de *explosive*) y objetos pequeños (mediana de la superficie de
  la etiqueta por imagen: 3,1 %).
- Sin revisión sistemática de rostros, matrículas o texto identificable en las 1.685 imágenes.

## Qué contiene `data/`

| Archivo | Contenido | Generado por |
|---|---|---|
| `tabla_maestra.csv`, `tabla_maestra_meta.json` | Resultados del benchmark y SHA-256 de sus entradas | `experiments/generar_tabla_maestra.py` |
| `eda_imagenes.csv`, `dice_por_caso.csv`, `dice_por_area.csv`, `limitacion_area.csv`, `calidad_shift.csv`, `calidad_resumen.json` | Derivadas livianas, sin imágenes | `experiments/generar_tablas_dashboard.py` |

## Cita obligatoria

Sharifi, A., Zibaei, A. y Rezaei, M. (2021). *A deep learning based hazardous materials (HAZMAT) sign detection robot
with restricted computational resources*. Machine Learning with Applications, 6, 100104.
