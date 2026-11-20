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
| Q5 v0.1 | Scopus | 38 | Conteo piloto; exportación falló. Omitida por el usuario el 18/11 |
| Q5 v0.1 | WoS Core Collection | 14 | Conteo piloto; exportación falló. Omitida por el usuario el 18/11 |
| Q6 v0.1 | Ambas | — | No ejecutada; omitida por el usuario el 18/11 |

Filtros: años 2019–2026; ninguno adicional de idioma o tipo documental. Q0 histórico sin filtro de año. Las cifras entre bases no se suman como estudios únicos. Q5 aún no se suma al corpus descargado. Q0–Q4 se descargaron el 17/11, según nombres y marcas horarias originales. La verificación y los intentos fallidos de exportación de Q5 continuaron el 18/11.

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
