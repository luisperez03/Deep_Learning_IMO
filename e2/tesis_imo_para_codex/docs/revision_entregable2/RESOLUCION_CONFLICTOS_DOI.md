# Resolución de conflictos DOI

**Estado 20/11/2026: adjudicados con el registro oficial; alias agregados al parser; fusión NO aplicada a los registros.** Las exportaciones crudas y `registros_conciliados.json` no están en el repositorio, así que `procesar_exportaciones.py` no se ha vuelto a correr. Los conteos de abajo son los **esperados** al correrlo; se confirman solo cuando se ejecute sobre los trece archivos.

Esta versión sustituye la del 18/11, que afirmaba sin URL que las dos variantes de PeerJ resolvían al mismo artículo y que `10.1504/ijes.2021.10039166` no era un DOI registrado. Ambas afirmaciones resultaron falsas al consultar el registro (ver conflictos 1 y 2).

## Método

`verificar_doi_registro.py` consulta, para cada DOI y sin seguir redirecciones:

- `https://api.crossref.org/works/<doi>`: registro de metadatos de los DOI editoriales.
- `https://api.datacite.org/dois/<doi>`: registro de los DOI de arXiv (`10.48550`).
- `https://doi.org/<doi>`: resolutor oficial; 302 con destino = DOI registrado; 404 = no existe.

La salida completa, con hora de consulta y metadatos, está en `evidencia_doi_registro.json`. Las páginas de PeerJ e Inderscience devolvieron 403 a la consulta automática; la adjudicación se apoya en Crossref y doi.org, que son el registro oficial del DOI.

## Conflicto 1: MCV-UNet (E2-3569 / E2-5674)

| DOI | Crossref | doi.org | Qué dice el registro |
|---|---|---|---|
| `10.7717/peerj-cs.2146` | 200 | 302 → `https://peerj.com/articles/cs-2146` | *MCV-UNet: a modified convolution & transformer hybrid encoder-decoder network…*; Xu y Wang; PeerJ Computer Science, vol. 10, e2146, 2024 |
| `10.7717/peerjcs.2146` | 404 | 404 | No está registrado |

**Adjudicación:** mismo artículo; canónico `10.7717/peerj-cs.2146`. La variante sin guion no existe como DOI: es un error de la exportación. Fusionar.

## Conflicto 2: Multi-level spatial attention network (E2-5639 / E2-5682)

| DOI | Crossref | doi.org | Qué dice el registro |
|---|---|---|---|
| `10.1504/ijes.2021.116134` | 200 | 302 → `http://www.inderscience.com/link.php?id=116134` | *Multi-level spatial attention network for image data segmentation*; Guo, Jiang y Jiang; International Journal of Embedded Systems, 14(3), 289, 2021 |
| `10.1504/ijes.2021.10039166` | 301 | 301 → `https://doi.org/10.1504/IJES.2021.116134` | DOI registrado como alias que redirige al canónico |

**Adjudicación:** mismo artículo; canónico `10.1504/ijes.2021.116134`. El otro sí es un DOI registrado (no un identificador interno, como decía la versión anterior), pero el propio registro lo redirige al canónico. Fusionar.

## Conflicto 3: Multiple-Scale Inpainting CNN for Retinal OCT (E2-5459 / E2-5878)

| DOI | Crossref | doi.org | Qué dice el registro |
|---|---|---|---|
| `10.3788/cjl202148.1507004` | 200 | 302 → `opticsjournal.net/.../zgjg/48/15/1507004` | Título original en chino; título en inglés *Multiple-Scale Inpainting Convolutional Neural Network for Retinal OCT Image Segmentation*; Yuan Kun y Huo Li; Chinese Journal of Lasers, 48(15), 1507004, 2021 |
| `10.3788/cjl202118.1507001` | 404 | 404 | No está registrado |
| `10.3788/cjl202148.1507001` (control) | 200 | 302 | Es otro artículo del mismo número (reconstrucción fotoacústica; primer autor Deng Kexin) |

**Adjudicación:** mismo artículo; canónico `10.3788/cjl202148.1507004`. La variante no existe como DOI. El control descarta que el número de artículo `1507001` apunte, por error de volumen, al registro de este título: corresponde a otro trabajo. La identidad se apoya en título y año iguales (la regla del parser) y en que el registro canónico tiene exactamente ese título en inglés. Fusionar.

## Tabla de alias y aplicación

Los tres alias están en `DOI_ALIAS` de `procesar_exportaciones.py`. Al correrlo:

- cada DOI se normaliza al canónico antes de agrupar; el DOI recibido queda en `doi_original` del registro;
- ningún ID se renumera: el grupo conserva el ID histórico del registro con DOI canónico, y el ID del grupo absorbido queda en `ids_absorbidos` (y en `ids_absorbidos` de `conciliacion.json`);
- `conciliacion.json` informa `alias_doi_aplicados` con los registros que pasaron por cada alias.

| Alias (como llega) | Canónico | Par de IDs |
|---|---|---|
| `10.7717/peerjcs.2146` | `10.7717/peerj-cs.2146` | E2-3569 / E2-5674 |
| `10.1504/ijes.2021.10039166` | `10.1504/ijes.2021.116134` | E2-5639 / E2-5682 |
| `10.3788/cjl202118.1507001` | `10.3788/cjl202148.1507004` | E2-5459 / E2-5878 |

Qué ID sobrevive en cada par lo decide la regla anterior (el del DOI canónico); el repositorio no guarda qué ID tenía cada DOI, así que no se fija aquí.

**Prueba:** con un fixture sintético de dos exportaciones (un título con DOI canónico y alias, más un duplicado ordinario), el parser original da 3 grupos y 1 ambigüedad; el modificado da 2 grupos, 0 ambigüedades, conserva `E2-0001` y registra `E2-0003` en `ids_absorbidos`. No se corrió sobre los datos reales.

## Conteo esperado

El parser crea exactamente un grupo extra por cada coincidencia ambigua, y `conciliacion.json` registra tres. Con los tres alias:

| | Antes | Esperado |
|---|---:|---:|
| Registros importados | 9.646 | 9.646 |
| Grupos (estudios únicos) | 6.325 | **6.322** |
| Duplicados conciliados | 3.321 | **3.324** |
| Coincidencias ambiguas | 3 | **0** |
| IDs históricos conservados para trazabilidad | 6.325 | 6.325 (3 en `ids_absorbidos`) |

Si al correrlo el resultado no es 6.322 / 3.324 / 0, hay otra diferencia en los datos y no debe ajustarse a mano.

El cribado automático (`resumen_cribado.json`, 6.325) se calculó antes de la fusión y debe recalcularse sobre los 6.322 grupos.
