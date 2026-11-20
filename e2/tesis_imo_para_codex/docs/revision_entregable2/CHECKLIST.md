# Checklist operativo de revisión E2

**20/11: control vigente en [CHECKLIST_ENTREGA_E2.md](CHECKLIST_ENTREGA_E2.md).** Las casillas históricas siguientes no certifican PRISMA cerrado, fusiones DOI aplicadas ni ranking bibliométrico verificado. Las 50 corridas y test sí están auditados; el usuario confirmó aceptación docente de diez modelos totales. Réplica y revisión sistemática siguen pendientes.

- [x] Revisar requisitos de literatura y estado del Entregable 1.
- [x] Preparar alcance, tres preguntas y criterios de revisión.
- [x] Preparar siete ecuaciones piloto, incluidas tres transversales.
- [x] Definir extracción, cribado, calidad y puntuación previa.
- [x] Localizar acceso institucional a Scopus y WoS.
- [x] Verificar ingreso a Scopus tras autenticación del usuario.
- [x] Verificar acceso a WoS Core Collection (Editions: All).
- [x] Ejecutar Q1 v0.2 en ambas bases y conservar exportaciones completas.
- [x] Conciliar Q1: 29 registros, 7 duplicados, 22 referencias únicas.
- [x] Leer títulos/resúmenes de Q1 y registrar sugerencias de cribado por separado.
- [x] Validar Q2 v0.1 en ambas bases y registrar sus conteos piloto.
- [x] Ejecutar y exportar Q0, Q2, Q3 y Q4 en ambas bases; verificar sumas de lotes.
- [x] Conservar IDs bibliográficos al ampliar la conciliación.
- [x] Preparar comparación inicial de siete candidatos y tabla transversal con fuentes primarias.
- [x] Registrar decisión del usuario del 18/11: omitir Q5 y Q6 y continuar con las siete candidatas propuestas.
- [x] Resolver tres coincidencias con DOI contradictorios: documentado en `RESOLUCION_CONFLICTOS_DOI.md`.
- [x] Ejecutar y exportar las consultas del alcance reducido Q0–Q4 en Scopus y WoS.
- [ ] Identificar ediciones específicas accesibles de WoS.
- [ ] Congelar versión definitiva tras piloto y registrar cambios.
- [x] Exportar los registros Q0–Q4 y verificar conteos.
- [x] Deduplicar y vincular versiones preservando procedencia: 9.646 registros, 3.321 duplicados, 6.325 grupos conciliados.
- [ ] Validar cribado y leer textos completos. Existe precribado automático de 6.325 registros, no elegibilidad definitiva.
- [ ] Validar con el investigador las decisiones de elegibilidad.
- [x] Completar extracción y valoración de calidad por estudio: fichas metodologicas de las 7 redes en `FICHAS_EXTRACCION_7REDES.md`.
- [ ] Verificar citas, año/categoría de cuartil y fuentes oficiales.
- [x] Puntuar candidatos y justificar excluidos relevantes: formalizado en `PUNTUACION_Y_EXCLUSIONES.md`.
- [ ] Registrar evidencia de confirmación docente de diez modelos totales y cuatro familias. La implementación v2 no equivale a esa confirmación.
- [x] Elegir paper guía y definir réplica y limitación medible: SegFormer-B0 con meta de 37,4 % mIoU ADE20K y adaptación binaria HAZMAT.
- [x] Completar tabla de mecanismos de otras modalidades: ampliado en `MECANISMOS_TRANSVERSALES.md`.
- [x] Preparar PRISMA con conteos conciliados: documento estructurado en `FLUJO_PRISMA.md`.
- [x] Auditar documentalmente variantes, código, licencias y rutas de adaptación de las siete redes: `VIABILIDAD_MODELOS.md`.
- [x] Proponer SegFormer-B0 como paper guía, separando reproducción de evaluación, entrenamiento original y adaptación HAZMAT.
- [ ] Registrar confirmación docente del alcance de réplica; la especificación por sí sola no la acredita.
- [x] Preparar rama v2 sin pesos, fuentes de autores fijadas por commit/hash y dependencias principales fijadas. No se descargaron checkpoints preentrenados.
- [ ] Ejecutar preflight GPU en Colab para las diez redes reales, incluido VM-UNet; no confundir prueba local CPU con validación CUDA.
- [x] Integrar síntesis crítica en Jupyter Book: capítulo `libro/literatura.md` compilado con éxito en HTML.

Este checklist corresponde a la revisión bibliográfica. No sustituye el checklist experimental completo de la página 63 de la guía. Las casillas pendientes no representan resultados negativos.

Q5 y Q6 se omiten por decisión explícita del usuario, no por ausencia de literatura. Q5 conserva conteos piloto sin exportación y Q6 no se ejecutó. No se declara completo el requisito original de tres búsquedas transversales. La selección de siete redes es una lista de trabajo aceptada, no un Top 7 obtenido del cribado completo.
