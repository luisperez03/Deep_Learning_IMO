# Revisión bibliográfica del Entregable 2

Versión 0.1, preparada el 17 de noviembre de 2026 (America/Bogota).
Actualización del 18/11: Q0–Q4 ejecutadas y exportadas en ambas bases. Por decisión explícita del usuario se omiten Q5 y Q6: Q5 conserva conteos piloto sin exportación; Q6 no se ejecutó. Las ecuaciones originales se conservan como historial. El requisito original de tres búsquedas transversales no se declara cumplido con este alcance reducido. Consultas exactas, cambios y conteos en `BITACORA_BUSQUEDAS.md`; esa bitácora prevalece sobre Q1 v0.1 como registro de la recuperación realizada.
Actualización del 22/11: la omisión de Q5 y Q6 se revocó; las dos se ejecutaron y exportaron en Scopus y WoS con las ecuaciones de abajo y los filtros 2019–2026 (conteos y archivos en `BITACORA_BUSQUEDAS.md`, sección "Reapertura de Q5 y Q6"). Las tres búsquedas transversales (Q4, Q5 y Q6) quedan ejecutadas.
Estado: piloto de búsqueda. Las ecuaciones se validan antes de cerrar la recuperación definitiva; cualquier cambio conserva versión y resultados anteriores. No es un preregistro retrospectivo del Entregable 1.

## Alcance y preguntas

Segmentación semántica binaria de rótulos de mercancías peligrosas en fotografías. Entrada RGB, salida máscara etiqueta/fondo. Separar antecedentes de detección, clasificación, OCR y segmentación; sus métricas no son intercambiables.

- RQ-L1: ¿Qué datos, anotaciones, modelos y protocolos se han utilizado para localizar y segmentar rótulos HAZMAT/IMDG?
- RQ-L2: ¿Qué arquitecturas de segmentación ofrecen mecanismos pertinentes para objetos pequeños, bordes y fondos complejos, con evidencia reproducible?
- RQ-L3: ¿Qué limitaciones metodológicas y mecanismos transferibles justifican las comparaciones del benchmark y una futura contribución?

Plan de diez modelos totales, incluidos FCN, SegNet y U-Net, sujeto a confirmación del solapamiento por el docente. El baseline trivial y el no profundo se registran aparte. No se declara una selección final de siete redes con la búsqueda exploratoria.

## Fuentes y ventana

Scopus y Web of Science Core Collection como bases principales; registrar índices y cobertura realmente accesibles. IEEE Xplore como complemento o sustitución documentada si una base no está disponible. arXiv y repositorios de autores son fuentes complementarias, no sustitutos de las dos bases principales.

Ventana principal: 2019 hasta la fecha real de ejecución en 2026. Los filtros por año no excluyen publicaciones futuras dentro del año: comprobar fecha de publicación de cada registro. Artículos anteriores se admiten como fundamentos por búsqueda de referencias o consulta identificada como histórica. Idiomas de lectura: inglés y español; no filtrar inicialmente por idioma para medir exclusiones. Artículos y conferencias originales; revisiones para rastrear fuentes, tesis para contexto. Preprints se distinguen de publicaciones revisadas por pares.

Q1 es una preferencia, no condición absoluta. Registrar fuente, categoría y año del cuartil; no asignar cuartiles a conferencias ni inventar valores. Separar evaluación de revista y calidad metodológica del estudio.

## Ecuaciones piloto

Aplicar cada núcleo en Scopus con TITLE-ABS-KEY(núcleo) AND PUBYEAR > 2018 AND PUBYEAR < 2027. En WoS con TS=(núcleo) AND PY=(2019-2026). Registrar la cadena exacta aceptada por cada plataforma. Los campos Topic y TITLE-ABS-KEY no tienen cobertura idéntica: declarar esa diferencia. Q0 es una comprobación histórica sin filtro de año.

### Q0: referencia del corpus, sin filtro temporal
```
HAZMAT13 OR "HAZMAT-13" OR DeepHAZMAT
```

### Q1: dominio
```
(hazmat OR "hazardous material*" OR "dangerous goods") AND (placard* OR label* OR sign*) AND ("deep learning" OR segmentation OR detection)
```

### Q2: tarea y dificultad
```
"semantic segmentation" AND ("small object*" OR "small target*" OR "boundary-aware" OR "cluttered background*") AND ("deep learning" OR transformer* OR convolution*)
```

### Q3: familias de segmentación
```
"semantic segmentation" AND ("encoder-decoder" OR transformer* OR "state space model*") AND (benchmark* OR compar* OR efficient*)
```

### Q4: mecanismo de atención desde lenguaje
```
("self-attention" OR transformer*) AND ("natural language processing" OR "machine translation") AND (transfer OR adaptation) AND (vision OR segmentation)
```

### Q5: mecanismos de estado desde secuencias
```
("state space model*" OR Mamba) AND ("language model*" OR "time series" OR "signal processing") AND (transfer OR adaptation) AND (vision OR segmentation)
```

### Q6: convoluciones dilatadas desde señales
```
("dilated convolution*" OR "temporal convolution*") AND (audio OR speech OR "time series") AND (transfer OR adaptation) AND (vision OR segmentation)
```

Si una ecuación transversal no recupera resultados, conservar el cero observado y ampliar en una versión posterior retirando el último bloque; no declarar ausencia del mecanismo. Las modalidades médicas son otros dominios de imagen, no una modalidad distinta: por sí solas no satisfacen este bloque transversal.

## Registro y recuperación

Por búsqueda: ID, versión, base/colección, ecuación exacta, fecha/hora local, filtros, número total mostrado, número exportado, formato, archivo y alcance. Guardar exportación original sin editar (RIS o BibTeX y, si es posible, metadatos completos con resumen, citas y referencias). Partir exportaciones grandes en rangos no solapados y verificar su suma. Un fallo de acceso no equivale a cero resultados.

Las consultas web abiertas y las búsquedas dirigidas por nombre se registran separadas de las consultas sistemáticas. No sumar resultados del buscador web a los conteos de Scopus/WoS. La lista de candidatos conocidos sirve para comprobar sensibilidad, no para restringir artificialmente la revisión a esos nombres.

## Cribado

1. Deduplicar DOI normalizado; después título normalizado, autor y año. Revisar manualmente coincidencias inciertas.
2. Conservar todos los identificadores y bases de origen. Vincular preprint y artículo final bajo un ID de estudio; una extensión con experimentos diferentes conserva su registro y relación.
3. Título/resumen: retener dominio directo o mecanismo de segmentación pertinente; ante duda pasar a texto completo.
4. Texto completo: clasificar como contexto, arquitectura candidata, mecanismo transferible o exclusión.
5. Registrar un motivo principal de exclusión: tarea sin transferencia justificable; sin método original (solo contexto); sin información suficiente para reproducir; duplicado/versionado; fuera de ventana sin fundamento; idioma no evaluable; texto completo no recuperado (separar de exclusión científica).
6. La revisión asistida por IA requiere validación del investigador. No calcular kappa entre respuestas del mismo asistente ni presentarlas como dos revisores independientes. Si existe segundo revisor humano, registrar su evaluación y acuerdo.

## Extracción por estudio

ID, procedencia de búsqueda, título, autores, año, DOI/URL, versión/publicación, alcance leído, tarea, dominio, dataset/licencia, unidad independiente, partición, riesgo de duplicados, arquitectura/familia, definición completa de capas, código/commit/licencia, pesos/corpus de preentrenamiento, pérdida, resolución, hiperparámetros, semillas, métricas y agregación, resultados con tabla/página, incertidumbre/pruebas, ajuste de baselines, parámetros/FLOPs/latencia/hardware, limitaciones y transferibilidad. Desconocido se registra como pendiente, no como ausente.

## Calidad y selección de modelos

Antes de puntuar, exigir salida de segmentación adaptable a RGB 2D, evidencia primaria accesible y posibilidad razonable de reproducción. Registrar los modelos que requieren prompts o anotaciones adicionales como otra condición experimental; no compararlos usando máscaras de test como prompts.

Puntuación propuesta (0-100), fijada antes de evaluar la lista completa:
- Pertinencia (25): 0 sin relación; 1 segmentación sin mecanismo específico; 2 mecanismo pertinente; 3 evaluación de dificultad compartida; 4 evidencia en rótulos o dominio comparable.
- Reproducibilidad (25): 0 insuficiente; 1 descripción parcial; 2 especificación suficiente; 3 código de autores y protocolo; 4 añade pesos/configuraciones/datos verificables. Puntuar solo evidencia comprobada.
- Evidencia experimental (20): 0 sin comparación; 1 métricas puntuales; 2 baselines y partición claros; 3 controles de equidad y repeticiones; 4 incertidumbre y control de dependencia adecuados.
- Viabilidad (10): 0 incompatible; 1 adaptación mayor; 2 adaptación viable con costo incierto; 3 configuración publicada plausible; 4 ejecución piloto dentro del presupuesto. Sin piloto no asignar 4.
- Recencia (10): 4 para 2025-2026; 3 para 2023-2024; 2 para 2021-2022; 1 para 2019-2020; 0 anteriores (excepción fundacional documentada).
- Citación normalizada por antigüedad (10): citas/max(1,2026-año+1), con misma base y fecha para todos; convertir al percentil entre candidatos elegibles y multiplicar por 10. Registrar conteo bruto y fuente. Si falta, mantener puntuación total pendiente y hacer sensibilidad sin este componente; no tratar faltante como cero.

Primeros cinco componentes: peso × puntuación/4. Registrar justificación por componente. No comparar directamente Dice de datasets diferentes ni equiparar frecuencia de citas con calidad.

Diversidad como restricción adicional: al menos cuatro familias defendibles (CNN, Transformer, híbrido CNN-Transformer, SSM, por ejemplo), con definiciones antes de seleccionar. Attention U-Net sigue siendo CNN con compuertas; el preentrenamiento no crea por sí solo otra arquitectura. Evitar contar variantes B0/B1 de una misma red como familias distintas. Elegir el conjunto mejor puntuado que cumpla diversidad y presupuesto; documentar cada sustitución. Empates: reproducibilidad, luego menor costo medido comparable; si no hay costo, dejar empate pendiente. Conservar FCN/SegNet/U-Net como referencias heredadas sin afirmar que ganaron esta selección bibliográfica.

## Paper guía y cierre

Elegirlo después del cribado: resultado replicable bajo datos/partición/métrica originales y limitación medible. Un antecedente HAZMAT puede ser referencia del dominio sin ser automáticamente el paper guía de arquitectura. Definir éxito de réplica antes de ejecutar. No descargar datasets ni entrenar en esta fase.

Cerrar cuando se hayan ejecutado las ecuaciones definitivas en dos bases, exportado y conciliado todos los registros, cribado el corpus, leído fuentes primarias candidatas, completado puntajes y documentado cuatro familias, paper guía y presupuesto. Emitir PRISMA con cifras conciliadas, tabla de excluidos, matriz de extracción y tabla de mecanismos; ninguno se considera completo hoy.

## Fuentes operativas

- Guía local: Entregable2DeepLearningProject.pdf, secciones 3.1-3.4, 5 y Apéndice B.
- Catálogo institucional observado: https://uninorte.libguides.com/az.php
- Scopus: https://uninorte.libguides.com/scopus
- WoS: https://uninorte.libguides.com/web_of_science
- Sintaxis Scopus: https://www.elsevier.support/scopus/answer/how-can-i-best-use-the-advanced-search
- Constructor WoS: https://webofscience.zendesk.com/hc/en-us/articles/20130361503249-Advanced-Search-Query-Builder
