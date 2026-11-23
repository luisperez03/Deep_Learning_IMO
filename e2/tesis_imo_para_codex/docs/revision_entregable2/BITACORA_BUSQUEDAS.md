# Bitácora de búsqueda: 17–18 de noviembre de 2026

Fecha local America/Bogota. Acceso institucional Uninorte; autenticación completada por el usuario. Scopus y Web of Science Core Collection funcionaron. En WoS se seleccionó Editions: All (las ediciones específicas accesibles aún no se inventarían ni se presuponen).

| Consulta | Base | Resultados observados | Exportación y estado |
|---|---|---:|---|
| Q1 v0.1 | Scopus | 578 | Piloto descartado como ecuación final; sin exportación |
| Q1 v0.2 | Scopus | 21 | RIS completo, información bibliográfica, resumen y palabras clave |
| Q1 v0.2 | WoS Core Collection | 8 | Texto plano, Full Record and Cited References |
| Q0 v0.1 | Scopus | 1 | RIS completo; recupera DeepHAZMAT |
| Q0 v0.1 | WoS Core Collection | 1 | Texto completo de registro y referencias; recupera DeepHAZMAT |
| Q2 v0.1 | Scopus | 1009 | RIS completo verificado |
| Q2 v0.1 | WoS Core Collection | 538 | Full Record and Cited References, lotes 1–500 y 501–538 verificados |
| Q3 v0.1 | Scopus | 5017 | RIS completo; descarga grande recuperada desde Exports |
| Q3 v0.1 | WoS Core Collection | 2592 | Full Record, lotes 1–1000, 1001–2000, 2001–2592; sin lista de referencias citadas |
| Q4 v0.1 | Scopus | 351 | RIS completo verificado |
| Q4 v0.1 | WoS Core Collection | 108 | Full Record and Cited References verificado |
| Q5 v0.1 | Scopus | 38 | 18/11: conteo piloto, exportación fallida. 22/11: reabierta; RIS completo (bibliográfica, resumen y palabras clave), 38 registros |
| Q5 v0.1 | WoS Core Collection | 14 | 18/11: conteo piloto, exportación fallida. 22/11: reabierta; Full Record and Cited References, 14 registros |
| Q6 v0.1 | Scopus | 11 | 22/11: primera ejecución; RIS completo, 11 registros |
| Q6 v0.1 | WoS Core Collection | 6 | 22/11: primera ejecución; Full Record and Cited References, 6 registros |

Filtros: años 2019–2026; ninguno adicional de idioma o tipo documental. Q0 histórico sin filtro de año. Las cifras entre bases no se suman como estudios únicos. Q5 y Q6 se reabrieron el 22/11 y se suman al corpus (ver "Reapertura de Q5 y Q6"). Q0–Q4 se descargaron el 17/11, según nombres y marcas horarias originales. La verificación y los intentos fallidos de exportación de Q5 continuaron el 18/11.

## Consultas exactas

Q1 v0.1, Scopus:

```text
TITLE-ABS-KEY((hazmat OR "hazardous material*" OR "dangerous goods") AND (placard* OR label* OR sign*) AND ("deep learning" OR segmentation OR detection)) AND PUBYEAR > 2018 AND PUBYEAR < 2027
```

Se revisó porque `sign*` y `detection` recuperaban señales y detección genérica de sustancias, alejadas del rótulo visual. No se cambió para favorecer una arquitectura.

Q1 v0.2, Scopus:

```text
TITLE-ABS-KEY((hazmat OR "hazardous material*" OR "dangerous goods") AND (placard* OR label OR labels OR sign OR signs) AND ("deep learning" OR segmentation OR "computer vision" OR "object detection" OR "sign detection" OR "label detection")) AND PUBYEAR > 2018 AND PUBYEAR < 2027
```

Q1 v0.2, WoS:

```text
TS=((hazmat OR "hazardous material*" OR "dangerous goods") AND (placard* OR label OR labels OR sign OR signs) AND ("deep learning" OR segmentation OR "computer vision" OR "object detection" OR "sign detection" OR "label detection")) AND PY=(2019-2026)
```

Q2 v0.1, Scopus:

```text
TITLE-ABS-KEY("semantic segmentation" AND ("small object*" OR "small target*" OR "boundary-aware" OR "cluttered background*") AND ("deep learning" OR transformer* OR convolution*)) AND PUBYEAR > 2018 AND PUBYEAR < 2027
```

Q2 v0.1, WoS:

```text
TS=("semantic segmentation" AND ("small object*" OR "small target*" OR "boundary-aware" OR "cluttered background*") AND ("deep learning" OR transformer* OR convolution*)) AND PY=(2019-2026)
```

## Evidencia conservada

- `exportaciones/scopus_Q1_v02_20261117.ris`: 21 registros.
- `exportaciones/wos_Q1_v02_20261117.txt`: 8 registros.
- `conciliacion.json`: conteos y SHA-256 de los archivos originales.
- `registros_conciliados.json`: metadatos originales, resúmenes y procedencia.
- `MATRIZ_CRIBADO.md`: inventario generado; las decisiones sugeridas están aparte en `LECTURA_INICIAL_Q1.md` para no perderlas al regenerar.

Resultado histórico de conciliación Q1: **29 registros importados, 7 duplicados, 22 referencias únicas, 0 coincidencias ambiguas**.

Actualización Q0–Q4: **9646 registros importados, 3321 duplicados conciliados y 6325 grupos de referencias**. Hay tres coincidencias de título/año con DOI contradictorios, preservadas por separado en `conciliacion.json`; el total único es provisional hasta resolverlas. La deduplicación usa DOI y luego título normalizado/año si no existen DOI contradictorios. El procesador ahora conserva los IDs previos y las decisiones guardadas. No es PRISMA final ni cribado de 6325 estudios.

Las consultas exactas Q0 y Q3–Q5 son los núcleos del protocolo sin modificaciones: `TITLE-ABS-KEY(núcleo) AND PUBYEAR > 2018 AND PUBYEAR < 2027` en Scopus y `TS=(núcleo) AND PY=(2019-2026)` en WoS; Q0 omite los filtros temporales.

Incidencia del 18/11: Scopus mostró error de exportación; WoS mostró fallo HTTP al guardar. Se refrescaron las pestañas y se reabrió WoS desde el acceso institucional por indicación del usuario. Posteriormente el usuario decidió omitir Q5/Q6: ya no se espera autenticación para estas descargas. Un fallo de exportación no es un resultado cero.

## Continuación concreta

1. Continuar con el corpus Q0–Q4 y las siete candidatas aceptadas como lista de trabajo; Q5/Q6 fuera del alcance por decisión del usuario.
2. Comprobar sensibilidad por búsqueda suplementaria de títulos/DOI de candidatos ausentes. No recortar el corpus a los primeros resultados por conveniencia.
3. Conciliar el corpus ampliado; validar el cribado sugerido y leer textos completos de los estudios que sobrevivan.
4. Extraer arquitectura, código, datos, evaluación y coste; verificar publicaciones, citas y cuartiles; después puntuar y escoger siete redes adicionales si se confirma el solapamiento.

Accesos estables: [Scopus vía Uninorte](https://uninorte.libguides.com/scopus), [Web of Science vía Uninorte](https://uninorte.libguides.com/web_of_science). No se guardaron contraseñas ni enlaces de autenticación en estos documentos.

## Reapertura de Q5 y Q6 (22/11/2026)

La omisión del 18/11 se revocó el 22/11 y Q5 y Q6 se ejecutaron y exportaron en las dos bases, con las ecuaciones del protocolo y los filtros de Q1 y Q2 (2019–2026; en WoS, Core Collection con Editions: All). Acceso institucional de Uninorte; la elección de cuenta y el inicio de sesión de Scopus los hizo el usuario.

Q5 v0.1, Scopus:

```text
TITLE-ABS-KEY(("state space model*" OR Mamba) AND ("language model*" OR "time series" OR "signal processing") AND (transfer OR adaptation) AND (vision OR segmentation)) AND PUBYEAR > 2018 AND PUBYEAR < 2027
```

Q5 v0.1, WoS:

```text
TS=(("state space model*" OR Mamba) AND ("language model*" OR "time series" OR "signal processing") AND (transfer OR adaptation) AND (vision OR segmentation)) AND PY=(2019-2026)
```

Q6 v0.1, Scopus:

```text
TITLE-ABS-KEY(("dilated convolution*" OR "temporal convolution*") AND (audio OR speech OR "time series") AND (transfer OR adaptation) AND (vision OR segmentation)) AND PUBYEAR > 2018 AND PUBYEAR < 2027
```

Q6 v0.1, WoS:

```text
TS=(("dilated convolution*" OR "temporal convolution*") AND (audio OR speech OR "time series") AND (transfer OR adaptation) AND (vision OR segmentation)) AND PY=(2019-2026)
```

| Archivo | Registros | Mostrados en la base | sha256 |
|---|---:|---:|---|
| `exportaciones/scopus_Q5_v01_20261122.ris` | 38 | 38 | `075151c128b5430af53e3c715aa1ea4d6d1658323a23a3259c10f189c4202752` |
| `exportaciones/wos_Q5_v01_20261122.txt` | 14 | 14 | `b0c3c6fdda92f1610f1b34a93346f0e2af0264b491ee114eb194ac840b3dca75` |
| `exportaciones/scopus_Q6_v01_20261122.ris` | 11 | 11 | `bf8dcb3471e0947d1d562e43ed96a77cc7071e9b7a4a38d4a3a9e7376e9c873a` |
| `exportaciones/wos_Q6_v01_20261122.txt` | 6 | 6 | `08956b6a941a2b22a6985b88a57c90921751a63e38699618f47b411fff0720ba` |

Los conteos de Q5 coinciden con los pilotos del 18/11 (38 y 14). **Por qué había fallado la exportación de Scopus:** Scopus genera el archivo, pero el enlace de descarga pasa por el host `exports-scopus-com` del proxy institucional, que responde 503. La descarga se completó desde el host original de exportaciones con el mismo enlace firmado que genera Scopus para ese trabajo, sin cambiar el contenido. Los archivos son RIS sin editar, con el mismo formato y los mismos campos que Q0–Q4.

Conciliación con las 17 exportaciones (`procesar_exportaciones.py`, `conciliacion.json`): **9.715 registros, 6.366 grupos, 3.349 duplicados, 0 coincidencias ambiguas**. Los 6.322 grupos anteriores conservan su ID; Q5 y Q6 aportan 53 grupos, 44 nuevos y 9 que ya estaban en el corpus. Cribado del bloque en `PRISMA_FINAL.md`.

