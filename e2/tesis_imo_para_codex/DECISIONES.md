# Bitácora de decisiones y preguntas

## 2026-11-20 — Integración E2 y revisión completa de requisitos

- El usuario pidió integrar todo y revisar cada requisito para entregar el domingo 22/11. Confirmó que el profesor acepta diez modelos totales (3 E1 + 7 nuevos). Esta confirmación no dispensa otros requisitos de la guía.
- Libro local actualizado: E2 como ruta principal, capítulos E1 conservados como antecedentes. Añadidas diez fichas/diagramas, método aplicado, resultados auditados, contraste principal, curvas, discusión y checklist de 76 controles más 21 casillas del apéndice B.
- Se ejecutó ampliación exploratoria posterior al test: intercambio pareado Monte Carlo, Wilcoxon-Holm, Friedman/Nemenyi, efectos con IC, rangos, Pareto puntual, estratos de área train y planificación MDD/R. No se presenta como preregistro. Friedman p≈7,17e-64 detecta heterogeneidad; TransUNet–FCN no demuestra mejora clara. Referencia vacía: Dice=0, exactitud≈0,9625.
- Corregida literatura: precribado automático no es elegibilidad definitiva; DOI no aplicados, bibliometría incompleta, FCN no es FCN-8s y U-Net compacta no comparte R18 con UNet++. El capítulo anterior se conserva en `docs/revision_entregable2/LITERATURA_ANTES_INTEGRACION_20261120.md`.
- Faltan experimentos centrales: réplica paper, clásico fuerte, búsqueda/ablaciones, calibración/robustez/interpretabilidad/incertidumbre predictiva, prueba formal de sobreajuste, techo/anotación y cierre bibliográfico/defensa. No retrofechar decisiones ni ajustar usando el test visto. Ver `CHECKLIST_ENTREGA_E2.md`.
- No se modificaron datos ni checkpoints, no se entrenó y no se publicó. La integración es revisión local, no entrega final científicamente completa ni garantía de nota.

## 2026-11-19 — Test recibido y auditado: benchmark E2 cerrado

- El usuario entregó `../resultados_test_e2_final/`. Se auditaron las 50 corridas, 10.900 registros por imagen y 2.180 máscaras de semilla 11, sin incidencias. Protocolo, firmas, hashes declarados de checkpoints, agregación e intervalos coinciden con el cierre previo. No se ejecutó entrenamiento ni nueva inferencia.
- Test contiene 218 imágenes y 122 grupos, sin máscaras vacías. Se verificaron hashes de imágenes/máscaras originales y se recalcularon métricas desde los PNG de semilla 11. Las otras semillas se verificaron desde sus registros, sin repetir inferencia.
- TransUNet lidera Dice test (0,86260); FCN 0,85354; U-Net 0,84637. Contraste principal TransUNet–FCN: +0,00906, IC95 pareado de grupos [-0,00828; 0,02779], con cinco semillas fijas. No demuestra mejora clara ni equivalencia. FCN es una referencia parsimoniosa por tamaño, no la ganadora en Dice.
- Entrenamiento en L4 y test en T4; no comparar tiempos como hardware idéntico ni reportarlos como latencia pura. Mantener exposición previa al test en E1 y límites de grupos/recetas en la discusión.
- Evidencia: `scripts/auditar_test_e2.py`, `resultados/auditoria_test_e2/AUDITORIA.json` y `CONCLUSIONES.md`. Próximo: integrar evidencia al entregable y resolver por separado réplica del paper guía y cierre bibliográfico. No afinar modelos con este test. No se subió ni desplegó nada.

## 2026-11-19 — Cierre autorizado y preparación de evaluación final E2

- Tras auditar las 50 corridas y proponer revisión cualitativa/cierre/test, el usuario indicó «dale procede». Esta autorización cubre la evaluación final propuesta; no volver a solicitarla sin cambio de alcance.
- Se revisaron seis casos dirigidos de validación, con inferencias CPU ilustrativas de FCN/U-Net/TransUNet, semilla 11. Panel y límites en `resultados/revision_visual_e2/`. No hay máscaras vacías en validación; no afirmar desempeño sobre imágenes enteramente negativas.
- Se congelaron las 50 selecciones de checkpoint mediante hashes en `config/protocolo_test_e2_cerrado.json`, sin cambiar modelos, recetas o umbral. Test 218 imágenes/122 grupos, batch de inferencia 1 y azar fijo para SegNeXt. No se usa test para selección posterior.
- Se preparó evaluador CUDA sin optimizador, reanudable por corrida, resultados por imagen/grupo/semilla, IC95 de grupos con semillas fijas y contrastes frente a FCN. Contraste principal TransUNet–FCN; demás exploratorios. Se declara exposición previa al test E1.
- Notebook `../Evaluacion_Final_E2_Colab.ipynb` y ZIP `../paquete_test_e2_cerrado.zip` separados del entrenamiento. Instalación incluye creación del entorno, setuptools y salida visible. Checkpoints se leen desde Drive; no se empaquetan ni alteran.
- Dos pruebas sintéticas del evaluador aprobadas (ponderación por grupo, contraste pareado e integridad); notebook compilable y ZIP verificado. Ejecución CUDA completa pendiente en Colab; no presentar resultados de test aún.

## 2026-11-19 — Auditoría de las 50 corridas E2 recibidas

- El usuario entregó `../tesis_imo_e2_arquitecturas_reales_v2/` y pidió revisión y próximos pasos. Se verificaron 50 corridas completas, diez modelos y semillas 11/23/37/53/71, sin incidencias en los controles ejecutados.
- Los 100 hashes de checkpoints best/last coinciden; los mejores checkpoints se leyeron con `weights_only=True`. Historias, fingerprint, época seleccionada, métricas y membresía de validación son coherentes. Validación: 218 imágenes y 122 grupos. Una sola configuración, identidad y entorno NVIDIA L4 en las corridas.
- La identidad de fuentes guardada coincide con la reconstruida desde el ZIP v2 entregado: `59a30d289ce7995b64b362d60e7859bf3445fa9e576e479abcca6b3f4ac77a62`. No usar el hash de scripts locales nuevos para desconocer la procedencia del paquete ejecutado.
- TransUNet lidera Dice de validación (media 0,85324, DE entre semillas 0,00638); U-Net 0,83200 y FCN 0,83092. Son resultados de checkpoints seleccionados por validación, no resultados de test ni prueba de superioridad universal.
- Evidencia reproducible: `scripts/auditar_resultados_e2.py`, `resultados/auditoria_e2_recibida/auditoria.json` e `INFORME.md`. No se modificaron las corridas entregadas ni se ejecutó inferencia de test.
- Propuesta siguiente: congelar las 50 recetas/checkpoints, revisar cualitativamente errores en validación y preparar evaluación final de test para todos los modelos/semillas. Ejecutarla tras cierre autorizado; declarar exposición previa al test E1. La réplica bibliográfica y PRISMA siguen separados.

## 2026-11-18 — Corrección autorizada del paquete Colab v2

- El usuario pidió revisar el trabajo nuevo y después continuar hasta dejar el archivo listo para Colab. Se autoriza preparación y validación; no se iniciaron las 50 corridas ni se subió nada a GitHub.
- Se identificaron prototipos mal denominados, diferencias runner/especificación y cierres bibliográficos no acreditados. Informe: `docs/revision_entregable2/REVISION_CAMBIOS_20261118.md`.
- Se sustituyeron por implementaciones reales: SMP 0.5.0 (DeepLabv3+/UNet++ R18 y SegFormer MiT-B0), PIDNet-S y TransUNet de autores, MSCAN-T/LightHamHead con infraestructura PyTorch mínima, y VSSM VM-UNet oficial con selective_scan CUDA. Commits, hashes y licencias en `third_party/e2`. No hay fallback a arquitecturas distintas ni carga de pesos externos.
- Configuración ejecutable: `config/experimento_e2.json`, recetas fijas por modelo, lote nominal 2 y cola de 3, pérdida binaria por imagen, auxiliares PIDNet explícitas. Se vuelven a ejecutar las bases bajo estas condiciones; no se modifica el E1 histórico. Este presupuesto no constituye tuning exhaustivo ni réplica de los artículos.
- Preflight exige integridad de train/val, separación de grupos y forward/backward/paso de optimizador con sintéticos. Runner exige esa validación para el código, datos, configuración y entorno actuales. Reanudación conserva RNG y mejor checkpoint asociado a una época; rechaza cambios de identidad.
- Entorno local nuevo `.venv-e2` para pruebas CPU, separado de `.venv-cpu`. VM-UNet y ejecución GPU quedan sujetos al preflight dentro de Colab; no afirmar validación CUDA desde el portátil.
- Documentación vigente de ejecución: `COLAB_E2_LEEME.md`. Paquete antiguo obsoleto; preparar `paquete_colab_e2_v2.zip` y notebook v2. Persisten pendientes bibliográficos, alcance docente y permiso antes de push/despliegue.
- Cierre local: nueve modelos reales pasaron forward/backward/paso de optimizador y evaluación sintéticos con lote 3 a 256; cuatro pruebas de contrato/reanudación pasaron y `pip check` no reporta incompatibilidades. ZIP v2 verificado (2522 archivos, aproximadamente 176,46 MiB), notebook compila y contiene el hash correcto. Evidencia: `resultados/validacion_e2_v2/VALIDACION_FINAL.json`. Pendiente preflight GPU en Colab, especialmente VM-UNet. No se ejecutó entrenamiento con datos reales.

## Decisiones confirmadas a la fecha
- El proyecto corresponde a una Maestría en Estadística Aplicada, no a una implementación comercial.
- Existe interés en localizar etiquetas y luego clasificarlas.
- Se quiere estudiar experimentalmente el funcionamiento matemático de las redes neuronales, incluyendo componentes desde cero.
- El profesor sugirió estudiar una formulación con segmentación.
- Las tres carpetas actuales son insumos de la fase de auditoría; **no se ha aprobado mezclarlas**.
- Los datos originales se preservan sin modificaciones.

## Hipótesis por comprobar (NO son hechos)
- El nombre `Hazmat Pacards.v5i.coco` podría indicar exportación COCO.
- Los otros datasets podrían aportar imágenes de contexto, máscaras o anotaciones útiles, pero no se conoce aún su contenido.
- Algunas clases posiblemente podrían mapearse a nomenclatura IMO/IMDG; esto exige inspeccionar etiquetas y ejemplos.

## Pendientes tras auditoría
- ¿Qué dataset realmente contiene etiquetas IMO y qué granularidad de clases?
- ¿Hay bounding boxes, polígonos, máscaras o solo clasificación a nivel de imagen?
- ¿Se puede establecer mapeo semántico fiel entre datasets?
- ¿Bastará una primera fase de detección o se requieren nuevas anotaciones para segmentación real?
- ¿Los datos son sintéticos, reales o mixtos? ¿Cómo se evitarán efectos de dominio?
- ¿Existen duplicados o archivos provenientes del mismo original en distintas particiones?
- ¿Qué licencias y restricciones de uso impone cada fuente?
- ¿Qué alcance y criterios de éxito se acordarán con el profesor?

## Registro de decisiones futuras
Usar: fecha, decisión, evidencia (ruta/informe), alternativas, justificación, impacto y aprobación del investigador.

## 2026-11-06 — Primer bloque reproducible autorizado

- **Autorización:** el investigador indicó «dale, empieza» tras la propuesta de
  definición, inventario, panel visual y primer cuaderno de EDA. Pidió reproducibilidad.
- **Alcance adoptado para este bloque:** VOC2012 como base inicial y segmentación
  binaria etiqueta/fondo como formulación propuesta. Sin entrenamiento ni mezcla.
- **Confirmado:** 1.685 JPG, 1.583 pares de imagen/máscara técnicamente válidos,
  102 imágenes sin máscara. No se encontraron duplicados exactos SHA256 entre JPG.
- **Confirmado:** 69 pares presentan discrepancias en conjuntos de clases XML/máscara;
  image_1018 no tiene XML. Esto exige revisión; no implica 69 errores espaciales.
- **Evidencia:** resultados/inicio_voc/inventario.csv, resumen.json y ejecucion.json;
  docs/REVISION_VISUAL_INICIAL.md; notebooks/01_eda_voc.ipynb.
- **Reproducibilidad:** configuración versionable, semilla, versiones fijadas,
  hashes de datos y código, generación ejecutable del cuaderno y pruebas propias.
- **Hipótesis:** segmentación binaria permite un comienzo viable pese a las
  discrepancias semánticas. La calidad espacial debe validarse por separado.
- **Pendiente:** casi duplicados, grupos de escena, procedencia/licencia, EDA
  exhaustivo, revisión bibliográfica, tres modelos, delta, test y protocolo final.
- El estado antiguo «sin inspeccionar» se conserva como registro histórico, no
  como descripción actual. El auditor general heredado aún requiere adaptación;
  el flujo focalizado se documenta en INICIO_REPRODUCIBLE.md.

## 2026-11-06 — EDA gráfico y segunda etapa

- **Autorización:** el investigador pidió más gráficos, explicaciones en el código
  y continuar con la siguiente etapa. Se amplió diagnóstico, no entrenamiento.
- **Confirmado:** 16 gráficos exportables y nueve celdas ejecutables comentadas.
  Búsqueda exhaustiva de similitud: 748 pares candidatos, 146 entre train/val.
  Sin grupos exactamente iguales en píxeles RGB. Fuente: resumen.json en revision_voc.
- **Observado por el asistente:** 12 pares de los paneles muestran composiciones
  muy semejantes entre particiones; seis máscaras prioritarias revisadas.
- **Alerta:** aparente omisión de un rótulo en image_1461. La tarea binaria tampoco
  debe asumir que toda máscara técnicamente válida es completa.
- **Decisión:** no considerar independientes los splits originales sin revisar
  similitud y escenas. `folder` XML es pista de origen, no identificador de escena.
- **Pendiente:** adjudicación de candidatos, reglas de cobertura y grupos finales,
  procedencia/licencia, literatura y diseño experimental. Evidencia y límites en
  docs/REVISION_ETAPA_2.md. No se alteraron originales ni se fijaron nuevos splits.

## 2026-11-06 — Preparación técnica previa al entrenamiento

- **Autorización:** el investigador pidió preparar todo hasta antes de correr
  modelos y validar primero su PC. No autoriza iniciar entrenamiento en esta etapa.
- **Preparado:** grupos conservadores con todos los enlaces perceptuales conocidos;
  cuarentena por grupo ante discrepancias semánticas, posible omisión en image_1461
  y cajas con cobertura de máscara inferior al 5 %. No se borran originales.
- **Resultado:** 1.451 pares elegibles por QC automático, 132 en cuarentena y
  102 sin máscara. 1.015/218/218 imágenes para train/val/test, con 567/122/122
  grupos. Cero enlaces candidatos conocidos entre las nuevas particiones.
- **Límite:** grupos de similitud no equivalen a escenas certificadas; no se
  adjudicaron manualmente los 748 pares ni todas las máscaras.
- **Reproducibilidad:** manifiestos y hashes, copias de máscara 0/1, normalización
  de train, loader diferido, métricas verificadas y test reservado por defecto.
- **Hardware observado:** Ryzen 5 7520U (4C/8T), 7,28 GiB visibles, Radeon integrada;
  aproximadamente 1 GiB libre en la consulta. No se midieron temperatura ni carga.
- **Presupuesto propuesto:** CPU 2 hilos, batch 1, 256×256, cero workers y una
  corrida a la vez. Falta aprobación y prueba mínima de memoria/tiempo.
- **Procedencia probable:** HAZMAT-13, apoyada por fuente de autores y coincidencia
  de 1.685 imágenes/13 clases. No se descargó para comparar hashes.
- **Pendiente académico:** FCN/SegNet/U-Net están fundamentados como familias, pero
  no demostrados como tres baselines recurrentes en HAZMAT. Selección y arquitecturas
  no se presentan como cerradas. Ver docs/PROCEDENCIA_Y_LITERATURA.md.
- **Punto de parada:** entrenamiento deshabilitado. Validar datos, delta, protocolo,
  selección bibliográfica y equipo antes de instalar/ejecutar el entorno de modelos.

## 2026-11-07 — prueba local autorizada

- El investigador confirmó que se aceptan baselines generales y que la entrega
  es el domingo 8; ver docs/ALCANCE_DOMINGO.md para las adaptaciones propuestas.
- Tras ofrecer Colab, pidió ejecutar aquí y autorizó preparar PyTorch CPU y
  una prueba breve de cinco pasos con el modelo más pequeño. No se autorizan
  automáticamente las quince corridas del piloto con esta prueba.
- Se crea `.venv-cpu`, independiente del entorno del EDA, con descarga desde
  los índices oficiales de PyTorch y PyPI. Versiones instaladas se registrarán
  en `requirements-cpu.lock.txt`; no usar dependencias sin versión para reproducir.
- Se mantiene la reserva de 2 GiB libres para la prueba a 256 píxeles. Una
  detención por memoria no demuestra incapacidad del CPU ni fallo del modelo.
- Los resultados reales y el estado (completado o detenido) se guardan en
  `resultados/preentrenamiento/carga_fcn.json`; no presentar una instalación
  exitosa como evidencia de entrenamiento exitoso.

## Evaluación final del piloto de tasa fija (7 de noviembre de 2026)

El investigador autorizó cerrar la configuración antes de evaluar el test.
Se conservan los tres modelos, las cinco semillas comunes, la tasa 0,001,
el umbral 0,5 y el checkpoint seleccionado por Dice de validación para cada
corrida. La tasa alternativa 0,0003 quedó sin explorar por presupuesto.
La comparación corresponde a segmentación binaria. Se excluye el padding.
El promedio principal da el mismo peso a cada grupo y luego a cada semilla.
Los intervalos descriptivos remuestrean grupos completos con las cinco
semillas fijas. La dispersión entre semillas se informa por separado.

## 2026-11-07 — Libro del Entregable 1

- Confirmado por la profesora: configuración cerrada a 0,001, máximo 30 épocas y paciencia 5. Cinco semillas comunes. Se descartan 0,0003 y 60 épocas para esta entrega.
- Confirmado desde los resultados guardados: FCN encabeza validación y test. Se adopta como referencia del benchmark.
- Decisión de análisis: IC percentiles por grupo con semillas fijas y contrastes bilaterales pareados por semilla con Holm. La magnitud práctica propuesta de 0,02 y la MMD de planificación responden a preguntas distintas.
- Limitaciones abiertas: recurrencia de las arquitecturas en HAZMAT sin demostrar; alcance de licencia para redistribución de imágenes pendiente de confirmar. El cuaderno Colab de figuras está preparado para inferencia cualitativa local al entorno remoto.

## Cierre del Entregable 1 tras contrastar la metodología del profesor

- Confirmado: auditoría de las 15 corridas y nueve inferencias cualitativas de los casos preseleccionados. La edición vigente contiene figuras, conteos de errores y fundamentos vinculados al código. Las métricas de las figuras coinciden con sus registros cerrados.
- Decisión: conservar FCN como referencia semántica binaria y el test como evaluación histórica. Las piezas de clasificación, anotación de instancias y construcción de nuevas máscaras se documentan como propuestas. El alcance y los límites constan en `docs/ALINEACION_PROFESOR.md`.
- Pendiente para otras etapas: validación por unidades de captura reales, segunda anotación y evidencia externa antes de afirmar generalización portuaria.

## 2026-11-18 — Revisión bibliográfica E2 y coordinación con GitHub

- Confirmado por el usuario: continuar investigación y descargas bibliográficas. El usuario completa personalmente la autenticación institucional/Elsevier. No se solicitan contraseñas por chat.
- Confirmado por el usuario: preguntar antes de subir cambios a GitHub o desplegar. Daniel podría haber realizado cambios remotos; inspeccionar el estado remoto y preparar una integración revisable antes de pedir autorización. No se ha realizado push ni despliegue en esta fase.
- Trabajo local: protocolo, bitácora, exportaciones originales, conciliación y comparación exploratoria en `docs/revision_entregable2/`.
- Hipótesis pendiente de confirmación docente: diez modelos totales incluyendo FCN/SegNet/U-Net, con siete adicionales y al menos cuatro familias. No presentar una selección definitiva mientras falte cribado y puntuación.
- Decisión metodológica: separar detección de cajas, segmentación clásica y segmentación neuronal. Comparar también condiciones de preentrenamiento; una adaptación a HAZMAT no equivale a réplica de un artículo en sus datos originales.
- Corrección conceptual: el uso de dilatación en WaveNet no demuestra una transferencia histórica audio→visión; revisar el requisito transversal con fuentes originales.

### 2026-11-18 — Preparación del entorno de entrenamiento Colab/VSCode con persistencia

- **Diseño experimental cerrado:** Documentado en [`ESPECIFICACION_EXPERIMENTO_E2.md`](file:///C:/Users/luisa/Downloads/DEEP%20LEARNING/tesis_imo_para_codex/docs/revision_entregable2/ESPECIFICACION_EXPERIMENTO_E2.md). Se fijan 10 arquitecturas, inicialización aleatoria en la rama principal, 5 semillas (11, 23, 37, 53, 71), resolución $256\times 256$, pérdida combinada BCEWithLogits + soft Dice sin padding y selección por validación agrupada.
- **Implementación de los 10 adaptadores:** En [`scripts/modelos_e2.py`](file:///C:/Users/luisa/Downloads/DEEP%20LEARNING/tesis_imo_para_codex/scripts/modelos_e2.py) se implementaron y verificaron los forward passes para FCN, SegNet, U-Net, DeepLabv3+, UNet++, PIDNet-S, SegNeXt-T, SegFormer-B0, TransUNet 2D y VM-UNet. Todos entregan tensores `(B, 1, 256, 256)`.
- **Runner resiliente ante cortes:** Creado [`scripts/experimento_e2_colab.py`](file:///C:/Users/luisa/Downloads/DEEP%20LEARNING/tesis_imo_para_codex/scripts/experimento_e2_colab.py) y cuaderno interactivo [`notebooks/benchmark_e2_colab.ipynb`](file:///C:/Users/luisa/Downloads/DEEP%20LEARNING/tesis_imo_para_codex/notebooks/benchmark_e2_colab.ipynb). Implementa guardado atómico por época (`last.tmp` -> `last.pt`), preservación de estados RNG, detección de checkpoints para reanudar automáticamente tras desconexiones y guardado en Google Drive.


- **Mecanismos transversales y 4 familias:** Se completó [`MECANISMOS_TRANSVERSALES.md`](file:///C:/Users/luisa/Downloads/DEEP%20LEARNING/tesis_imo_para_codex/docs/revision_entregable2/MECANISMOS_TRANSVERSALES.md) fundamentando matemáticamente la transferencia hacia segmentación binaria (Self-Attention, SSM continuo, ASPP dilatado, control PID, atención convolucional MSCAN y puentes híbridos) abarcando 4 familias: CNN, Transformer, Híbrido y SSM.
- **Puntuación y exclusiones justificadas:** Se formalizó [`PUNTUACION_Y_EXCLUSIONES.md`](file:///C:/Users/luisa/Downloads/DEEP%20LEARNING/tesis_imo_para_codex/docs/revision_entregable2/PUNTUACION_Y_EXCLUSIONES.md) aplicando la rúbrica cuantitativa a las 7 candidatas (PIDNet-S: 88,89; SegFormer-B0: 79,17; UNet++: 76,39; DeepLabv3+: 73,61; SegNeXt-T: 69,44; VM-UNet: 63,89; TransUNet 2D: 61,11) y justificando el descarte formal de Attention U-Net, Swin-UNet, SAM/SAM-2 y YOLOv8-seg.
- **Compilación de Jupyter Book exitosa:** Se actualizó [`libro/literatura.md`](file:///C:/Users/luisa/Downloads/DEEP%20LEARNING/tesis_imo_para_codex/libro/literatura.md) integrando el flujo PRISMA 2020, las 4 familias y la auditoría de los 10 modelos. Se compiló el libro en HTML de manera limpia y sin errores (`libro/_build/html/index.html`).


- **Conflictos DOI resueltos:** Se documentó en `RESOLUCION_CONFLICTOS_DOI.md` el origen de las 3 discrepancias (PeerJ-CS con/sin guión, Inderscience ID interno vs DOI, y volumen/número de Chinese Journal of Lasers). Se adoptaron los DOIs canónicos de Crossref sin alterar IDs de grupo.
- **Cribado algorítmico ejecutado:** Se procesaron los 6.325 registros únicos en `resumen_cribado.json` y `cribado_automatico.json`. 2.186 registros retenidos (20 dominio HAZMAT, 507 arquitecturas directas, 1.659 mecanismos frontera), 2.052 excluidos (dominio no transferible, no experimentales, solo detección) y 2.087 en duda pendientes de inspección dirigida.
- **Extracción de las 7 candidatas:** Generado `FICHAS_EXTRACCION_7REDES.md` completando método, optimizador, pérdidas, hardware y vías de adaptación binaria.
- **Flujo PRISMA 2020 consolidado:** Documentado en `FLUJO_PRISMA.md` con conteos exactos (9.646 importados, 3.321 duplicados, 6.325 analizados).


- Confirmado: «vamos con esas, omite q5 y 6». Continuar con DeepLabv3+, UNet++, PIDNet-S, SegNeXt-T, SegFormer-B0, TransUNet 2D y VM-UNet como lista de trabajo; omitir la recuperación de Q5 y la ejecución de Q6.
- Conservar Q5 como consulta piloto con conteos 38/14 y exportación fallida; Q6 como no ejecutada. El corpus descargado sigue siendo Q0–Q4. No presentar ausencia de exportación como cero resultados.
- La lista aceptada requiere verificación de versiones, reproducibilidad y viabilidad; no es un ranking definitivo derivado del cribado de todo el corpus. El requisito original de búsquedas transversales permanece incompleto.
- Ya no se requiere resolver el inicio de sesión para Q5/Q6. Se mantiene la instrucción de preguntar antes de subir a GitHub o desplegar.

### Auditoría documental y propuesta experimental de las siete redes

- Realizado localmente: `docs/revision_entregable2/VIABILIDAD_MODELOS.md`, con artículos, código, licencias, variantes, pesos anunciados y riesgos de adaptación. No se instalaron modelos ni se descargaron pesos/datasets ni se entrenó en esta fase.
- Propuesta técnica: DeepLabv3+ ResNet-18; UNet++ ResNet-18 sin supervisión profunda (adaptación SMP); PIDNet-S; SegNeXt-T; SegFormer-B0; TransUNet 2D R50-ViT-B/16; VM-UNet base. Variantes propuestas, no resultados validados.
- Propuesta de paper guía: SegFormer-B0. Separar reproducción de evaluación ADE20K, réplica de entrenamiento y benchmark binario HAZMAT. Confirmar alcance aceptable con el docente.
- Condición: VM-UNet requiere prueba en entorno Linux/CUDA compatible; TransUNet necesita presupuesto medido. Hardware local histórico Windows/AMD no acredita esa compatibilidad. No sustituir SSM por CNN sin revisar cobertura de familias.
- Propuesta metodológica: comparar inicialización aleatoria por separado de pesos externos; conservar E1 histórico y volver a correr bases si cambia el presupuesto de selección. El test existente sigue congelado, declarando exposición previa y sin usarlo para elegir configuraciones.
- Pendientes: commits, entorno reproducible, condiciones/hashes de checkpoints, pruebas de adaptadores y mediciones de recursos. La consulta documental no certifica ejecución. No se efectuó subida ni despliegue; se mantiene la autorización previa exigida por el usuario.

## 2026-11-22 — Reapertura de Q5 y Q6

- Se revoca la omisión de Q5 y Q6 del 18/11 (sección anterior: «vamos con esas, omite q5 y 6»). Instrucción del 22/11: «Sí, reábrela y corre Q5 y Q6».
- Ejecutadas y exportadas en Scopus y WoS con las ecuaciones del protocolo y los filtros 2019–2026: Q5 38 + 14 registros (iguales a los conteos piloto) y Q6 11 + 6. Archivos, sha256 y la causa del fallo de exportación del 18/11 en `docs/revision_entregable2/BITACORA_BUSQUEDAS.md`, sección "Reapertura de Q5 y Q6".
- Corpus: 9.715 registros, 6.366 grupos y 3.349 duplicados (antes 9.646 / 6.322 / 3.324); los ID anteriores no cambian. Los 52 grupos del bloque transversal que no estaban revisados se criban completos (censo): 9 incluidos como mecanismo transferible y 43 excluidos con motivo (`PRISMA_FINAL.md`).
- Queda sin efecto lo anterior sobre que no se requiere iniciar sesión para Q5/Q6 y que el requisito de tres búsquedas transversales no se cumple: Q4, Q5 y Q6 están ejecutadas y exportadas.

