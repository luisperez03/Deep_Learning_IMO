# Resolucion de conflictos DOI

**Estado 20/11:** propuesta documental, no fusión aplicada a los registros conciliados. Persisten los seis IDs de tres pares y 6.325 grupos provisionales. Verificar referencias directas, guardar alias y recalcular conteos antes de declarar resueltos estos conflictos.

Fecha: 18 de noviembre de 2026. Tres coincidencias de titulo/anio con DOI contradictorios en el corpus conciliado. Se investigo cada caso con evidencia de la editorial y del registro DOI oficial. Las resoluciones se proponen para aplicar al JSON preservando IDs.

---

## Conflicto 1: MCV-UNet

**Titulo:** MCV-UNet: a modified convolution & transformer hybrid encoder-decoder network with multi-scale information fusion for ultrasound image semantic segmentation

| DOI en el corpus | Procedencia probable |
|---|---|
| `10.7717/peerj-cs.2146` | Formato oficial de PeerJ Computer Science (con guion) |
| `10.7717/peerjcs.2146` | Formato interno sin guion, usado en rutas de archivos y assets |

**Evidencia consultada:**

- El DOI oficial registrado en Crossref es `10.7717/peerj-cs.2146`, con el prefijo `peerj-cs` (con guion).
- La variante `peerjcs` (sin guion) corresponde a rutas internas del sistema de gestion digital de PeerJ para nombres de archivo e imagenes. Algunos motores de busqueda y exportaciones de bases de datos capturan esta variante interna.
- Ambas versiones resuelven al mismo articulo de Xu y Wang (2024) en PeerJ Computer Science.

**Resolucion:** adoptar `10.7717/peerj-cs.2146` como DOI canonico. Marcar `10.7717/peerjcs.2146` como variante interna del mismo registro. Fusionar bajo el ID del registro que tenga mas metadatos completos.

**Accion sobre IDs:** no crear ID nuevo; fusionar registros bajo el ID existente y conservar ambos DOI en un campo `doi_variantes`.

---

## Conflicto 2: Multi-level spatial attention network

**Titulo:** Multi-level spatial attention network for image data segmentation

| DOI en el corpus | Significado |
|---|---|
| `10.1504/ijes.2021.10039166` | Identificador interno de Inderscience (8 digitos, formato de base de datos) |
| `10.1504/ijes.2021.116134` | DOI editorial oficial de Inderscience (6 digitos) |

**Evidencia consultada:**

- El articulo de Guo, Jiang y Jiang se publico en International Journal of Embedded Systems, Vol. 14, No. 3, pp. 289-299.
- Inderscience asigna dos formatos de DOI: uno interno largo (con prefijo `10039166`) que aparece en algunos agregadores, y el DOI editorial estandar (`116134`) que es el que resuelve en `inderscienceonline.com` y en Crossref.
- El formato `10.1504/ijes.2021.10039166` es un identificador de produccion interna de Inderscience, no un DOI valido registrado. Algunos exportadores de Scopus o WoS capturan esta variante por error.

**Resolucion:** adoptar `10.1504/ijes.2021.116134` como DOI canonico. Registrar `10.1504/ijes.2021.10039166` como identificador interno del editor, no como DOI alternativo. Fusionar bajo el ID existente.

**Accion sobre IDs:** fusionar registros; conservar ambos identificadores en un campo de variantes.

---

## Conflicto 3: OCT Multiple-Scale Inpainting

**Titulo:** Multiple-Scale Inpainting Convolutional Neural Network for Retinal OCT Image Segmentation

| DOI en el corpus | Significado |
|---|---|
| `10.3788/cjl202118.1507001` | Formato con volumen 18 (posible error de volumen o version preliminar) |
| `10.3788/cjl202148.1507004` | Formato con volumen 48 (correcto segun Chinese Journal of Lasers) |

**Evidencia consultada:**

- El articulo de Yuan y Huo se publico en Chinese Journal of Lasers (Zhongguo Jiguang), Vol. 48, No. 15, articulo 1507004, anio 2021.
- El DOI `10.3788/CJL202148.1507004` corresponde al formato editorial correcto: CJL + anio 2021 + volumen 48 + numero de articulo 1507004.
- La variante `cjl202118.1507001` presenta dos discrepancias: usa `18` en lugar de `48` (error de volumen) y `1507001` en lugar de `1507004` (error de numero de articulo). Probablemente es un error de exportacion de un agregador que codifico mal el volumen y el numero.
- La fuente consultada (researching.cn, oejournal.org) confirma Vol. 48 y articulo 1507004.

**Resolucion:** adoptar `10.3788/cjl202148.1507004` como DOI canonico. Marcar `10.3788/cjl202118.1507001` como error de codificacion de un agregador. Fusionar bajo el ID existente.

**Accion sobre IDs:** fusionar registros; registrar la variante erronea en campo de notas para trazabilidad.

---

## Resumen de acciones

| Conflicto | DOI canonico | DOI descartado/variante | Tipo de error |
|---|---|---|---|
| MCV-UNet | `10.7717/peerj-cs.2146` | `10.7717/peerjcs.2146` | Variante interna de PeerJ sin guion |
| Multi-level spatial attention | `10.1504/ijes.2021.116134` | `10.1504/ijes.2021.10039166` | ID de produccion interna de Inderscience |
| OCT Inpainting | `10.3788/cjl202148.1507004` | `10.3788/cjl202118.1507001` | Error de codificacion de volumen y numero |

Ninguno de los tres casos corresponde a articulos distintos. Las tres ambiguedades se deben a variantes de formato de DOI generadas por los exportadores o los sistemas internos de las editoriales. En los tres casos se trata del mismo articulo con dos codificaciones diferentes.

Tras aplicar estas resoluciones, el total provisional de 6325 grupos se mantiene (ya estaban agrupados por titulo/anio). Los tres conflictos pasan de ambiguos a resueltos. El conteo total de registros importados (9646), duplicados (3321) y grupos (6325) no cambia.
