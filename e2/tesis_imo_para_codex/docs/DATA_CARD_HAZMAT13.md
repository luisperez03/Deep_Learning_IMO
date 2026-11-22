# Data Card: HAZMAT-13 (rótulos de materiales peligrosos)

> **Trazabilidad.** Redactada el 2026-11-22. Fuentes: páginas públicas de los autores leídas en esa fecha (enlaces
> abajo) y la copia local, comprobada con `scripts/verificar_corpus_hazmat13.py`. No se descargó el conjunto
> alojado por los autores ni se redistribuye ninguna imagen. Lo que no se pudo comprobar está marcado como
> **no verificado**. Complementa a `docs/PROCEDENCIA_Y_LITERATURA.md` (registro de la búsqueda del 6 de noviembre).

## 1. Identidad y procedencia

| Campo | Valor |
|---|---|
| Nombre | HAZMAT-13, *Hazardous Materials Signs Dataset* |
| Autores | Laboratorio de Robótica Móvil Avanzada (MRL-AMRL), Universidad Azad de Qazvin, Irán; con coautoría de la Universidad de Leeds, Reino Unido |
| Repositorio | [github.com/mrl-amrl/HAZMAT13](https://github.com/mrl-amrl/HAZMAT13) (rama `main`; 5 commits, del 2 al 12 de julio de 2021; sin *releases* ni *tags*) |
| Artículo | Sharifi, Zibaei y Rezaei (2021), *A deep learning based hazardous materials (HAZMAT) sign detection robot with restricted computational resources*, Machine Learning with Applications, 100104, [doi:10.1016/j.mlwa.2021.100104](https://doi.org/10.1016/j.mlwa.2021.100104). Versión previa: [arXiv:2007.06392](https://arxiv.org/abs/2007.06392) (*DeepHAZMAT*) |
| Alojamiento de las imágenes | Tres carpetas de Google Drive enlazadas desde el README; en GitHub solo hay imágenes de ilustración |
| Versión publicada que corresponde a este corpus | «Original model»: 1.685 imágenes sin aumento, con cajas y máscaras. No existe número de versión: la referencia estable es el estado del repositorio en el commit `ec85498` (12 de julio de 2021) más la carpeta de Drive de esa fecha |
| Versiones que **no** se usan | «Augmented model» (4.065 imágenes por clase, 52.845 en total según el README; el artículo dice 52.835) y «Different distance» (conjunto pequeño a 50, 100 y 150 cm) |

### Cotejo de la copia local con lo publicado

| Rasgo | Publicado por los autores | Copia local (`datos/hazmat13_drive/VOC2012`) | Resultado |
|---|---|---|---|
| Imágenes originales | 1.685 | 1.685 JPG (`image_0000` a `image_1684`) | coincide |
| Clases | 13, con estos nombres y orden: poison, oxygen, flammable (gas), flammable-solid, corrosive, dangerous, non-flammable-gas, organic-peroxide, explosive, radioactive, inhalation-hazard, spontaneously-combustible, infectious-substance | las mismas 13, en el mismo orden (`label.json`, `labelmap.txt`) | coincide |
| Formato | Pascal VOC, con caja y máscara | XML de cajas (`bboxes/annotations`) y PNG de máscara (`SegmentationClass`, VOC) | coincide |
| Origen del etiquetado | laboratorio AMRL | cada XML conserva la ruta `/home/amrl/Desktop/<carpeta>/…`: `amrl` es el nombre del laboratorio; siete carpetas de trabajo, una por anotador | consistente |
| Cobertura de máscaras | el README dice que cada imagen lleva caja y máscara | 1.583 máscaras y 1.582 XML: 102 imágenes sin máscara, 103 sin XML y 1 máscara sin XML (`image_1018`) | **difiere**: no se sabe si la diferencia está en la carpeta original de Drive o se produjo aguas abajo |
| Metadatos EXIF | no descritos | ausentes en las 1.685 imágenes (sin cámara, fecha ni GPS) | no permite reconstruir la adquisición |
| Campos `source` y `database` de los XML | no descritos | `database = Unknown` en los 1.582 XML | sin información de origen por imagen |

**Veredicto sobre la identidad.** La copia local es, con muy alta probabilidad, la versión «Original model» de HAZMAT-13:
coinciden el conteo exacto, las 13 clases con su orden, el formato y la ruta del laboratorio en los XML. **No se
certifica igualdad binaria**: los autores no publican sumas de verificación y la carpeta de Drive no se descargó
para compararla. Para cerrar esa verificación hace falta descargar la carpeta «Original model» y correr
`scripts/verificar_corpus_hazmat13.py` sobre ella (sección 6); el paso requiere autorización expresa porque implica
descargar el conjunto.

### Huellas de la copia local (versión del corpus usada)

- Suma agregada de las 1.685 imágenes (SHA256 sobre nombre y SHA256 de cada archivo, orden alfabético):
  `58b938d42de5f4fb514d0b734554fc4a4ec6463e3840d959f995e0f2aeaf9f3b`.
- Suma agregada de los 1.582 XML (SHA256 sobre nombre y bytes): `090ab361d20d38e3d9191614c30581d48a70771e40bc03090d7925015996518f`.
- SHA256 por imagen y por máscara original de los 1.451 pares elegibles: `resultados/preentrenamiento/{train,val,test}.csv`
  (columnas `image_sha256`, `original_mask_sha256`); recomprobados el 2026-11-22: 2.902 de 2.902 coinciden con los archivos locales.
- Particiones por grupos (semilla 20261106): entrenamiento 1.015, validación 218, prueba 218; excluidas 234 (102 sin
  máscara y 132 en cuarentena por discrepancias semánticas o revisión manual). Detalle en `resultados/preentrenamiento/freeze_manifest.json`.

## 2. Contenido

- 1.685 fotografías de rótulos de materiales peligrosos (JPG, 13 tamaños distintos; los más comunes 388×520, 749×1000 y 485×1000).
- 2.639 cajas en 1.582 XML. Cajas por clase: explosive 406, radioactive 281, oxygen 256, inhalation-hazard 250,
  flammable-solid 216, dangerous 203, non-flammable-gas 185, flammable 174, organic-peroxide 163, corrosive 152,
  infectious-substance 149, spontaneously-combustible 125, poison 79.
- Condiciones descritas por los autores: ángulos, distorsiones, iluminación y fondos diversos; parte de las muestras proviene de situaciones de competencias RoboCup de años anteriores.
- Uso en este trabajo: segmentación binaria del rótulo (unión de las máscaras de las 13 clases). Los XML no entran en el entrenamiento.

## 3. Adquisición

Lo que se sabe, con su fuente:

- El README y el artículo describen las imágenes como tomadas con distintos ángulos, iluminación y fondos, y mencionan una parte de «competencias RoboCup de años anteriores». **No dicen** quién las fotografió, con qué cámara, en qué fechas ni si hay imágenes de internet o de terceros.
- El artículo menciona que el conjunto puede etiquetarse con labelImg y que el formato es Pascal VOC; no afirma de manera explícita que esa fuera la herramienta empleada. No documenta el número de anotadores, un protocolo de etiquetado, ni acuerdo entre anotadores.
- El robot descrito (ARKA, cámara Intel RealSense RGB-D) se menciona como fuente de un conjunto secundario futuro; el texto no dice que esas imágenes estén entre las 1.685.

Lo que **no es posible reconstruir** desde la copia: fecha, sensor, lugar o sesión de cada imagen (sin EXIF, `source`/`database` = `Unknown`).
Consecuencia metodológica ya aplicada en el proyecto: sin metadatos de sesión se agrupan por similitud perceptual para
evitar fugas (grupos de casi duplicados) y se declara que esa agrupación no sustituye la independencia por escena.

## 4. Licencia

- **Repositorio:** licencia MIT, *Copyright (c) 2021 MRL Advanced Mobile Robotics Laboratory* ([LICENSE](https://raw.githubusercontent.com/mrl-amrl/HAZMAT13/main/LICENSE)). El texto concede permiso para usar, copiar, modificar y distribuir «the Software» («Permission is hereby granted, free of charge»), sujeto a conservar el aviso de copyright.
- **Imágenes y anotaciones:** la licencia nombra «Software»; ni el README ni el artículo declaran una licencia de datos ni la titularidad de las fotografías. El artículo declara que el conjunto se publica abiertamente y el README pide citarlo, pero ninguno fija términos de uso.
- **Terceros:** parte de las imágenes parece provenir de situaciones de competencias; su titularidad y permisos no están documentados.

**Estado:** la licencia MIT del repositorio **no se puede extender con certeza a las imágenes**. Mientras no haya aclaración
de los autores: (a) el uso para investigación y docencia, con atribución y cita, es el uso razonable que el propio
repositorio invita; (b) este proyecto no redistribuye imágenes, máscaras ni cajas (están ignoradas por git; el libro solo
muestra figuras derivadas mínimas con su cita); (c) cualquier redistribución de las imágenes requiere escribir a los
autores primero. No se envió ningún mensaje a los autores.

**Cita obligatoria al usar el conjunto** (la pide el README): Sharifi, A., Zibaei, A. y Rezaei, M. (2021),
*A deep learning based hazardous materials (HAZMAT) sign detection robot with restricted computational resources*,
Machine Learning with Applications, 100104. doi:10.1016/j.mlwa.2021.100104.

## 5. Geografía, ética y limitaciones

- **Geografía:** no documentada por imagen. Los autores son iraníes (Qazvin) y la competencia RoboCup y un campo de pruebas en Sídney aparecen en el artículo, pero no se dice dónde se tomaron las 1.685. No se debe afirmar cobertura de ninguna región ni de puertos o terminales: el test mide generalización interna, no una operación portuaria independiente.
- **Personas y datos personales:** son fotografías de rótulos, no de personas, pero no se hizo una revisión sistemática de rostros, matrículas o texto identificable en las 1.685 (la revisión visual cubrió una muestra de 30 y casos puntuales). No se redistribuyen imágenes; si se publicaran, habría que revisarlas antes.
- **Anotación:** sin protocolo ni acuerdo entre anotadores publicados. El proyecto midió el ruido de etiqueta en 100 imágenes (rótulo cortado por el borde sin máscara, tipo falta) y un techo por segunda anotación independiente de 50 imágenes de validación (Dice 0,941); ver el capítulo `complementos_datos_e2`.
- **Cobertura de máscaras:** 102 imágenes sin máscara y 69 discrepancias entre clases del XML y de la máscara; se excluyeron o se pusieron en cuarentena (sección 1).
- **Clases desbalanceadas** (79 cajas de *poison* frente a 406 de *explosive*) y objetos pequeños (mediana de superficie de etiqueta por imagen: 3,1 %; fondo: 95,6 % de los píxeles).
- **Alcance de uso:** investigación académica sobre segmentación del rótulo. No es un conjunto validado para decisiones de seguridad ni de operación logística.

## 6. Instrucciones de obtención y verificación

1. Abrir el [repositorio de los autores](https://github.com/mrl-amrl/HAZMAT13) y leer el README vigente (puede cambiar; este documento refleja el estado del 2026-11-22).
2. Descargar la carpeta de Google Drive de «Original model» (enlace en el README, sección *Original model*). **Descargar el conjunto requiere autorización expresa del titular del proyecto**; no se hace de forma automática.
3. Dejar la estructura Pascal VOC así (la que usa este proyecto):

   ```text
   VOC2012/
     JPEGImages/            1.685 JPG
     SegmentationClass/     1.583 PNG
     bboxes/annotations/    1.582 XML (+ label.json, hazmat.names)
     ImageSets/Segmentation/{train,val,trainval}.txt
     labelmap.txt
   ```

   Los directorios `SegmentationClassRaw` y `SegmentationObject` están en la copia local (1.583 archivos cada uno) y no entran en el verificador; su procedencia no se comprobó.
4. Verificar la copia: `python scripts/verificar_corpus_hazmat13.py RUTA/VOC2012`. Debe terminar con «Coincide con la Data Card». Si las huellas difieren, la copia no es la usada aquí: no mezclar resultados.
5. Regenerar las particiones y el manifiesto con el flujo del proyecto (`docs/PREENTRENAMIENTO.md`, `PREENTRENAMIENTO_REPRODUCIBLE.md`); sus CSV traen el SHA256 por archivo para auditar cada imagen y máscara.
6. Citar el artículo y conservar el aviso de la licencia MIT del repositorio junto a cualquier derivado de código.

Enlaces de verificación, consultados el 2026-11-22: [README](https://github.com/mrl-amrl/HAZMAT13), [LICENSE](https://raw.githubusercontent.com/mrl-amrl/HAZMAT13/main/LICENSE), [historial de commits](https://github.com/mrl-amrl/HAZMAT13/commits/main), [arXiv:2007.06392](https://arxiv.org/abs/2007.06392).

## 7. Qué falta para cerrar del todo

| Pendiente | Qué lo cierra |
|---|---|
| Igualdad binaria con la carpeta de Drive | descargar «Original model» (con permiso) y correr el verificador |
| Licencia de las imágenes y titularidad de las partes de competencias | respuesta escrita de los autores (no enviada) |
| Cómo y cuándo se tomaron las fotografías | respuesta de los autores; no está en el repositorio ni en el artículo |
| Diferencia 102/103 imágenes sin máscara o XML frente a «cada imagen lleva máscara» | comparar con la carpeta original de Drive |
