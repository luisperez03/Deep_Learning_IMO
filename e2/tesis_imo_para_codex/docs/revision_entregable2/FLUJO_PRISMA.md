# Declaracion de flujo PRISMA 2020: Revision bibliografica Entregable 2

**RECTIFICACIÓN 20/11:** borrador histórico no cerrado. Los 2.052 excluidos son sugerencias automáticas; diez redes no equivalen a diez estudios elegibles de un cribado exhaustivo. Persisten 6.325 grupos provisionales y tres pares DOI sin fusión aplicada. Estado corregido: `libro/literatura.md` y `CHECKLIST_ENTREGA_E2.md`. No presentar lo siguiente como PRISMA final.

Fecha: 18 de noviembre de 2026.
Ambito: Seleccion y viabilidad de arquitecturas para segmentacion semantica binaria etiqueta/fondo de rotulos HAZMAT/IMDG.
Protocolo de origen: `docs/revision_entregable2/PROTOCOLO_BUSQUEDA.md`.

---

## 1. Fase de identificacion

### 1.1 Registros identificados en bases de datos indexadas

| Base de datos | Consultas ejecutadas | Total registros brutos |
|---|---|---:|
| Scopus | Q0 (1), Q1 v0.2 (21), Q2 (1009), Q3 (5017), Q4 (351) | 6.399 |
| Web of Science Core Collection | Q0 (1), Q1 v0.2 (8), Q2 (538), Q3 (2592), Q4 (108) | 3.247 |
| **Total importado de bases** | **13 archivos exportados (Q0-Q4)** | **9.646** |

*Nota de alcance:* Las consultas transversales Q5 (conteos piloto 38 Scopus / 14 WoS) y Q6 (no ejecutada) fueron omitidas por decision explicita del usuario tras fallo de sesion institucional. Los conteos de Q5 no se suman a los registros importados al no disponerse de exportacion integra.

### 1.2 Depuracion inicial de duplicados

- Registros importados brutos: **9.646**
- Duplicados tecnicos conciliados (mismo DOI o titulo/anio identico sin contradiccion): **3.321**
- Registros unicos iniciales pre-resolucion de discrepancias DOI: **6.325**
- Discrepancias DOI detectadas y resueltas documentalmente: **3**
  - MCV-UNet (`10.7717/peerj-cs.2146` oficial frente a `10.7717/peerjcs.2146` interno)
  - Multi-level spatial attention (`10.1504/ijes.2021.116134` oficial frente a `10.1504/ijes.2021.10039166` interno)
  - OCT Inpainting (`10.3788/cjl202148.1507004` oficial frente a `10.3788/cjl202118.1507001` error volumen/articulo)
- Registros unicos consolidados para cribado: **6.322** (o **6.325** manteniendo los grupos ID historicos con vinculacion cruzada)

---

## 2. Fase de cribado (Screening)

### 2.1 Cribado por titulo y resumen

Sobre los 6.325 registros unicos analizados mediante reglas semanticas y lexico controlado de tarea/mecanismo:

| Categoria | Registros | Proporcion (%) | Accion |
|---|---:|---:|---|
| Excluidos por tipo documental no experimental (actas/proceedings, errata, editoriales) | 72 | 1,14 % | Excluidos de plano |
| Excluidos por dominio no transferible (medicina sin arquitectura candidata, biologia, audio, etc.) | 1.732 | 27,38 % | Excluidos de plano |
| Excluidos por tarea sin transferencia (solo deteccion de cajas, clasificacion pura) | 248 | 3,92 % | Excluidos de plano |
| **Total excluidos en cribado inicial** | **2.052** | **32,44 %** | Fin de evaluacion |
| Registros retenidos por dominio directo HAZMAT / mercancias peligrosas (RQ-L1) | 20 | 0,32 % | Pasan a evaluacion de elegibilidad |
| Registros retenidos por arquitectura candidata evaluada en segmentacion (RQ-L2) | 507 | 8,02 % | Pasan a evaluacion de elegibilidad |
| Registros retenidos por mecanismo transferible (objetos pequenos, bordes, atencion, SSM) | 1.659 | 26,23 % | Pasan a evaluacion de elegibilidad |
| Registros en zona de duda (requieren inspeccion dirigida de texto completo) | 2.087 | 32,99 % | Pendientes de inspeccion secundaria |

---

## 3. Fase de elegibilidad (Eligibility)

### 3.1 Identificacion de candidatos de arquitectura (RQ-L2)

Presencia de arquitecturas clave en el corpus cribado:
- SegFormer / MiT: 549 menciones
- DeepLabv3 / DeepLabv3+: 661 menciones
- SegNet: 295 menciones
- TransUNet: 209 menciones
- FCN: 171 menciones
- UNet++ / U-Net++: 170 menciones
- SegNeXt / MSCAN: 51 menciones
- VM-UNet / Visual Mamba: 14 menciones
- PIDNet: 13 menciones

### 3.2 Articulos evaluados a texto completo para las 7 candidatas

Se realizo auditoria documental y extraccion metodologica de los 7 modelos seleccionados para conformar el benchmark de 10 arquitecturas (junto a FCN, SegNet y U-Net del E1):

1. DeepLabv3+ (Chen et al., ECCV 2018) - Familia CNN (ASPP)
2. UNet++ (Zhou et al., TMI 2020) - Familia CNN (conexiones densas)
3. PIDNet-S (Xu et al., CVPR 2023) - Familia CNN (tres ramas / control PID)
4. SegNeXt-T (Guo et al., NeurIPS 2022) - Familia CNN con atencion convolucional
5. SegFormer-B0 (Xie et al., NeurIPS 2021) - Familia Transformer jerarquico (Paper guia)
6. TransUNet 2D R50-ViT-B/16 (Chen et al., preprint 2021) - Familia Hibrida CNN-Transformer
7. VM-UNet (Ruan y Xiang, preprint 2024) - Familia SSM (State Space Model)

---

## 4. Fase de inclusion

- Estudios incluidos en sintesis cualitativa de dominio (RQ-L1): 10 articulos de Q1 priorizados para texto completo (`LECTURA_INICIAL_Q1.md`).
- Modelos incluidos en diseno experimental del benchmark (RQ-L2): 10 modelos (3 baselines E1 + 7 candidatos auditados documentalmente en `VIABILIDAD_MODELOS.md` y `FICHAS_EXTRACCION_7REDES.md`).
- Paper guia seleccionado para replica: SegFormer-B0 (NeurIPS 2021), meta de evaluacion mIoU 37,4 % en ADE20K y adaptacion HAZMAT binaria.

---

## 5. Diagrama de flujo estructurado

```text
[ IDENTIFICACION ]
  Registros identificados en Scopus (Q0-Q4): n = 6.399
  Registros identificados en WoS (Q0-Q4):    n = 3.247
       │
       ▼
  Total registros importados: n = 9.646
       │
       ▼
  Duplicados eliminados por conciliacion automatica: n = 3.321
  Resolucion documental de discrepancias DOI: n = 3
       │
       ▼
[ CRIBADO ]
  Registros unicos sometidos a cribado por titulo/resumen: n = 6.325
       │
       ├─────────────────────────────────────────┐
       ▼                                         ▼
  Registros excluidos (n = 2.052):          Registros preseleccionados:
    - Dominio no transferible: n = 1.732      - Dominio HAZMAT (RQ-L1): n = 20
    - Tarea sin segmentacion:  n = 248        - Arquitectura directa:   n = 507
    - Tipo no experimental:    n = 72         - Mecanismos frontera:    n = 1.659
                                              - Zona de duda:           n = 2.087
                                                 │
                                                 ▼
[ ELEGIBILIDAD ]
  Articulos candidatos priorizados para auditoria tecnica: n = 7
  Evaluacion metodologica, licencias, codigo y pesos: n = 7
       │
       ▼
[ INCLUSION ]
  Modelos fijados en propuesta de benchmark E2: n = 10
    - Baselines E1: FCN, SegNet, U-Net (n = 3)
    - Nuevas arquitecturas multimodales/multiescala (n = 7):
        DeepLabv3+, UNet++, PIDNet-S, SegNeXt-T,
        SegFormer-B0 (Paper guia), TransUNet 2D, VM-UNet
```
