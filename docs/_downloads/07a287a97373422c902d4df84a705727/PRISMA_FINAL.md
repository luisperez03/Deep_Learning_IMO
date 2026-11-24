# PRISMA del núcleo de dominio y error del precribado

Generado por `cribado_nucleo.py cerrar` desde `cribado_manual.csv` (decisiones en `decisiones_cribado_muestra.json` y `decisiones_cribado_nucleo.json`; criterios en `CRITERIOS_ELEGIBILIDAD.md`). Decisiones propuestas por un asistente de IA con criterios fijos; auditoría de consistencia en `auditoria_cribado.json`. Sin kappa: no hay dos revisores humanos. Precribado: `precribado_reconstruido.py`.

```text
Identificados: 9715  ->  duplicados: 3349  ->  grupos: 6366
Precribado automático: {'RETENER_DOMINIO': 23, 'EXCLUIR_TIPO': 134, 'EXCLUIR_DOMINIO': 2152, 'RETENER_MECANISMO': 3323, 'RETENER_ARQUITECTURA': 343, 'DUDA': 122, 'EXCLUIR_DETECCION': 68, 'EXCLUIR_TAREA': 201}
Núcleo de dominio revisado: 25 (texto completo leído en 2; el resto con el resumen, ver nota)  ->  incluidos: 10
Excluidos con motivo: {'tarea sin transferencia': 13, 'sin método original': 2}
Transversales Q5/Q6 no revisados antes (censo, título y resumen): 52  ->  incluidos: 9; excluidos con motivo: {'sin método original': 22, 'tarea sin transferencia': 21}
```

Discrepancia: en EXCLUIR, registro que se incluiría; en RETENER, registro que se excluiría; en DUDA, registro incluible.

| Estrato del precribado | Revisados | Discrepancias | Tasa [IC95 Wilson] | Registros del estrato | Discrepancias esperadas | Incluibles estimados [IC95] |
|---|---:|---:|---|---:|---|---|
| EXCLUIR | 60 | 22 | 0.3667 [0.2562, 0.4932] | 2504 | [642, 1235] | 918 [642, 1235] |
| DUDA | 60 | 40 | 0.6667 [0.5406, 0.7727] | 122 | [66, 94] | 81 [66, 94] |
| RETENER_MECANISMO | 40 | 18 | 0.45 [0.3071, 0.6017] | 3321 | [1020, 1998] | 1827 [1323, 2301] |
| RETENER_ARQUITECTURA | 40 | 8 | 0.2 [0.105, 0.3476] | 342 | [36, 119] | 274 [223, 306] |

Incluibles a título y resumen fuera del núcleo, estimados por estratos: 3100 (suma de los IC95 por estrato: 2254–3936; intervalo conservador, no simultáneo). No se evaluaron a texto completo: quedan fuera del alcance de la síntesis, que se limita al núcleo de dominio y a las arquitecturas puntuadas.

**Gap.** Denominador: 10 estudios de dominio incluidos; segmentan el rótulo por píxel con red profunda: 0; comparan arquitecturas con varias semillas: 0. Refutadores: ninguno.
