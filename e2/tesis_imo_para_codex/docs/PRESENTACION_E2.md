# Presentación oral del Entregable 2: guion y defensa

> **Trazabilidad.** Escrito el 2026-11-20 a partir de `resultados/auditoria_test_e2/CONCLUSIONES.md`, del checklist
> maestro, de `libro/complementos_datos_e2.md` y de las guías de los complementos. Las cifras del benchmark son las
> del test cerrado y auditado; lo que dice **[si se corrió]** se llena con los resúmenes de los cuadernos
> complementarios o se presenta como pendiente. Unos 15 minutos, 13 diapositivas.

## 1. El problema

Segmentar el rótulo de mercancía peligrosa contra el fondo, en HAZMAT-13. No es clasificar la clase IMO: es
delimitar el rombo. Una foto con su máscara.

## 2. Los datos y la unidad

1.451 fotos utilizables, 1.015 / 218 / 218, en 567 / 122 / 122 grupos de similitud. El grupo es la unidad: fotos
casi iguales no son observaciones independientes. Sin máscaras vacías: no se mide la falsa alarma en escenas sin
rótulos.

## 3. Lo que el EDA cambió

Ruido de etiqueta: 7 de 100 en revisión manual (IC95 3,4–13,8 %), todos rótulos cortados por el borde: parece una
regla de anotación del conjunto, no descuido. Sin cambio de distribución entre particiones (KS, MMD y
clasificador adversarial). Entropía, formas y tono por clase.

## 4. Fuga

Hashes y dHash sin cruces. Con embeddings: los grupos dHash no separan sesiones de fotos; 9 grupos de test tienen un
casi duplicado en train. Sin ellos el Dice baja 0,003 a 0,010 y el orden no cambia. **Defensa:** no se rehizo la
partición porque rompería la comparación con el E1 y el test ya estaba expuesto; se cuantifica.

## 5. Literatura y elección de las redes

Scopus y WoS, 9.646 registros, 6.322 estudios únicos tras deduplicar y adjudicar tres pares de DOI con Crossref
(el conteo final se confirma al correr el parser sobre las exportaciones). Diez redes en cuatro familias: CNN (FCN, SegNet,
U-Net, DeepLabv3+, UNet++, PIDNet-S), CNN con atención (SegNeXt-T), Transformer (SegFormer-B0), híbrida
(TransUNet) y espacio de estados (VM-UNet). Paper guía: SegFormer-B0.

## 6. El protocolo

Todas desde cero, 256 px, lote 2, 30 épocas con paciencia 5, cinco semillas, pérdida BCE + Dice. Elección por
validación; umbral 0,5 fijado antes del test. **Defensa:** desde cero para no mezclar arquitectura con datos
externos.

## 7. La tabla principal

| Red | Dice test ± DE semillas | IC95 por grupos |
|---|---:|---:|
| TransUNet | 0,863 ± 0,006 | 0,836–0,885 |
| FCN | 0,854 ± 0,014 | 0,828–0,874 |
| U-Net | 0,846 ± 0,014 | 0,818–0,870 |
| VM-UNet | 0,839 ± 0,030 | 0,810–0,863 |
| … | | |
| SegFormer-B0 | 0,705 ± 0,010 | 0,661–0,744 |
| Baseline clásico (gradient boosting) | 0,645 | 0,605–0,682 |

## 8. El contraste principal

TransUNet − FCN = 0,009, IC95 pareado por grupos de −0,008 a 0,028: incluye el cero. No hay superioridad ni
equivalencia demostradas con δ = 0,02. FCN tiene 72 mil parámetros frente a 105 millones: por parsimonia es
defendible, sin declararla ganadora en Dice.

## 9. Ranking y costo

Friedman y Nemenyi con los grupos como bloques; diagrama de diferencia crítica; frente de Pareto Dice contra
parámetros.

## 10. Diagnósticos

Calibración (Brier, ECE, temperatura), robustez con tres severidades, Integrated Gradients con fidelidad y
estabilidad, e incertidumbre conforme por clase en el top 3. **[si se corrió]**; si no, el protocolo preparado y
por qué falta.

## 11. Ablación y controles de sanidad

TransUNet sin Transformer, sin conexiones de salto y con BCE sola; permutación de etiquetas; búsqueda con el mismo
presupuesto. **[si se corrió]**

## 12. Réplica del paper guía

SegFormer-B0 en ADE20K, escala única, contra 37,4 % mIoU, con el criterio fijado antes: 37,43 % (IC95 35,47–38,30),
D = +0,03 puntos. Veredicto `compatible`: la cifra coincide, pero el IC de D sale del margen de ±1 punto. El
preprocesador por defecto de Hugging Face resta 1,5 puntos. Es réplica de evaluación, no de entrenamiento. Y la
limitación: el SegFormer-B0 de HAZMAT queda por debajo en todos los cuartiles de área; parece un déficit de nivel,
no de tamaño.

## 13. Amenazas a la validez y lo que no se hizo

El preregistro integral no es anterior al entrenamiento y la potencia no fue prospectiva (se dice, no se
retrofecha). El test se vio en el E1. Recetas distintas por red y sin búsqueda sistemática en el benchmark. Sin
segunda anotación, no hay techo humano medido. Un solo conjunto de datos.

## Preguntas que conviene llevar respondidas

- **¿Por qué TransUNet no "gana"?** Porque el IC de la diferencia con FCN incluye el cero; la media mayor no basta.
- **¿Por qué SegFormer, el paper guía, queda último?** Entrenado desde cero con una receta fija y lote 2; el artículo
  usa preentrenamiento en ImageNet y otra escala. No prueba que los Transformers sean peores en general.
- **¿Por qué el grupo y no la imagen?** Para no inflar n con fotos casi idénticas; las pruebas y los IC remuestrean
  grupos.
- **¿Son válidos los resultados si el test se vio en el E1?** Los pesos y el umbral se fijaron antes de esta
  evaluación y no se eligió nada con ella; se declara que no es un conjunto nunca observado.
- **¿Hay fuga?** Se auditó con hashes, dHash y embeddings; el efecto de los casi duplicados se cuantifica y no
  cambia el orden.
- **¿Qué es el techo de desempeño?** El acuerdo entre dos anotadores; sin segunda anotación no se inventa: se
  reporta como limitación, con el kit listo.
