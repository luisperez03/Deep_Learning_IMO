# Referencias aplicadas al problema y gap refutable

> **22/11/2026: cribado del núcleo cerrado.** Del corpus de 6.322 grupos se incluyeron 10 estudios sobre rótulos o placas de mercancía peligrosa. Ninguno segmenta el rótulo por píxel con una red ni compara arquitecturas con semillas (`PRISMA_FINAL.md`, `decisiones_cribado_nucleo.json`). El denominador del gap es de 13 estudios (esos 10 y tres anteriores a 2019) y la condición de abajo queda resuelta; redacción vigente en `libro/literatura.md`, sección "Gap y límites de la síntesis".

**20/11/2026.** DOI verificados en Crossref y doi.org con `verificar_doi_registro.py` (`evidencia_doi_registro.json`). Lo que hace cada estudio se tomó de su resumen, recuperado de OpenAlex a partir del DOI; **no se leyó el texto completo**. BibTeX en `libro/referencias_e2.bib`.

## Estudios aplicados a rótulos de mercancía peligrosa

| Referencia | DOI | Tarea según el resumen | Método | Relación con este trabajo |
|---|---|---|---|---|
| Sharifi, Zibaei y Rezaei (2021), *Machine Learning with Applications* 6 | `10.1016/j.mlwa.2021.100104` | Detección y segmentación de señales HAZMAT en un robot de rescate con CPU limitada | Detector tipo YOLO; la segmentación la hace GrabCut adaptativo con posproceso morfológico | Autores del conjunto HAZMAT-13; su segmentación no es una red profunda por píxel. El preprint `10.48550/arXiv.2007.06392` es el mismo trabajo |
| Zhang, Bahrami, Feng y Liu (2024), *IEEE Transactions on Artificial Intelligence* 5(4) | `10.1109/TAI.2023.3298588` | Detección y reconocimiento de placas en transporte de mercancía peligrosa, con clases desbalanceadas y no vistas | Detector profundo de cero disparos; fusión de información visual y textual; entropía cruzada con peso logarítmico | Mismo objeto (placas en vehículos), tarea de cajas y clases, no de máscara |
| Mohamed, Tunnermann y Mertsching (2018), IEEE SSRR | `10.1109/SSRR.2018.8468639` | Detección de etiquetas HAZMAT para robots de rescate; publica un conjunto de datos | Preselección por atención visual (saliencia) y clasificación con SIFT | Usa el contraste y el color saturado de la etiqueta, que es lo que debe aprender la red |
| Edlinger, Zauner y Zauner (2019), *Electronic Imaging* 31(7) | `10.2352/ISSN.2470-1173.2019.7.IRIACV-463` | Reconocimiento y localización de etiquetas HAZMAT en un robot de rescate | Compara visión clásica con detectores profundos con hardware limitado | Mide el efecto del fondo y de la iluminación, dos fuentes de error de HAZMAT-13 |
| Zhao, Parra y Delp (2013), IEEE GlobalSIP | `10.1109/GlobalSIP.2013.6736996` | Detección y reconocimiento de señales HAZMAT en el móvil | Mapas de saliencia y detector de cuadriláteros convexos | Línea base sin aprendizaje profundo; distorsión proyectiva, desenfoque y sombra |
| Thongkor, Parra Pozo, Zhao, Amornraksa y Delp (2016), *Electronic Imaging* 28(11) | `10.2352/ISSN.2470-1173.2016.11.IMAWM-459` | Localización de señales HAZMAT | Descriptores de forma de Fourier (magnitud y fase) tras recorte de canales de color y binarización | Contorno del rombo; robustez a baja resolución y perspectiva |

Seis estudios con DOI editorial verificado, más el preprint de DeepHAZMAT. Cumple el mínimo de tres referencias aplicadas (fila 02 del checklist).

## Gap refutable

**Afirmación.** En los seis estudios aplicados identificados (2013-2024), la tarea es detección, reconocimiento o localización del rótulo. Ninguno entrena una red profunda de segmentación semántica por píxel del rótulo, y ninguno compara varias arquitecturas con varias semillas bajo un presupuesto común. El único que produce una máscara (Sharifi et al., 2021) la obtiene con GrabCut sobre la caja del detector.

**Qué la refuta.** Un solo estudio revisado por pares que segmente por píxel rótulos o placas de mercancía peligrosa con una red profunda, o que compare arquitecturas de segmentación sobre ese objeto reportando la variabilidad entre corridas.

**Alcance y condición.** El denominador es este conjunto de seis, no el corpus de la revisión. Del corpus (6.322 grupos esperados tras los alias; ver `RESOLUCION_CONFLICTOS_DOI.md`), el precribado automático marcó 20 registros como `RETENER_DOMINIO` y 2.087 como duda, y ninguno se ha cribado a texto completo; las exportaciones no están en el repositorio. Por eso, hasta cerrar ese cribado, la redacción admisible es:

> En los estudios aplicados a rótulos de mercancía peligrosa que se identificaron, no se encontró ninguno que segmente el rótulo por píxel con una red profunda ni que compare arquitecturas con varias semillas. La conclusión queda condicionada a la revisión a texto completo de los registros de dominio del corpus.

No se escribe "ningún estudio previo" ni se da una proporción de estudios con datos o código: ambas cosas necesitan el denominador de estudios incluidos, que no está cerrado.

**Cómo se cierra.** Revisar a texto completo los 20 registros `RETENER_DOMINIO` (y las dudas que mencionen hazmat, placard, dangerous goods o label) después de correr `procesar_exportaciones.py` con los alias. Si ninguno refuta la afirmación, se quita la condición; si alguno la refuta, el gap se reformula sobre lo que ese estudio no hizo.
