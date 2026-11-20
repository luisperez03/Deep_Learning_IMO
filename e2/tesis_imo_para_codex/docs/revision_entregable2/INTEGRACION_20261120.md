# Integración local verificada — 20/11/2026

Petición: integrar la evidencia para E2 y revisar cada requisito; entrega domingo 22/11. El usuario confirma aceptación docente de diez modelos totales. No hay autorización para publicar en GitHub ni desplegar.

## Artefactos

- Libro: `libro/_build/html/index.html` desde la raíz del proyecto.
- Checklist: `docs/revision_entregable2/CHECKLIST_ENTREGA_E2.md`.
- Paquete portátil de revisión: `../Entregable2_integrado_revision_local.zip`; extraer y abrir `libro_html/index.html`.
- Evidencia de verificación: `resultados/integracion_e2/VERIFICACION.json` y `reproduccion.log`.

## Validación ejecutada

- Cuatro pruebas estadísticas aprobadas: Holm con referencia conocida, límite exacto de cinco pares, efecto cero y simetría de signos.
- Flujo `reproducir_entregable2.py` ejecutado de principio a fin: análisis, figuras, integración y compilación Jupyter Book con advertencias tratadas como errores.
- 598 referencias locales comprobadas sin archivos faltantes; resumen de 167 palabras; diez diagramas.
- 76 controles: 23 cubiertos, 32 parciales, 17 faltantes, 3 desviaciones históricas y 1 no aplicable. No son porcentajes de nota; cada fila agrupa requisitos de distinto peso.
- 21 casillas del apéndice B examinadas individualmente.
- ZIP verificado sin errores CRC; 13.020.323 bytes, SHA-256 `266cb834903d8556988cea5796b86ee43572371c6be2bae6f2a64f27bacea37f`.
- Revisión visual de figuras de curvas de validación, Pareto, rangos y esquema de arquitectura; sin recorte de etiquetas observado.

## Límite del cierre

Se integró la evidencia disponible; no se completaron por escritura los experimentos ausentes. El documento aún no debe presentarse como entrega final completa. Prioridades: complemento GPU sobre pesos congelados, réplica original, clásico fuerte/ablaciones, bibliografía/anotación y defensa. No se cambiaron originales ni pesos y no hubo publicación.
