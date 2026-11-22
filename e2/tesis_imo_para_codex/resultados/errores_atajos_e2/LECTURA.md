# Errores, subgrupos, daño potencial y atajos: lectura

> **Trazabilidad.** 22/11/2026. Protocolo escrito antes de calcular: `PROTOCOLO_ERRORES_ATAJOS_20261122.md` y
> `config/errores_atajos_e2_20261122.json` (SHA-256 en cada salida). Exploratorio: el test ya estaba observado; nada de
> esto cambia modelos, umbral ni checkpoints. Validación (218 imágenes, 122 grupos), 10 redes × 5 semillas; test solo
> como descripción secundaria. IC95 por bootstrap de grupos (4.000).
>
> **Hecho en local** (sin predicciones): `python scripts/errores_atajos_e2.py atributos --voc RUTA/VOC2012`, `extraer
> --diagnosticos RUTA/diag_descarga` (los `*_clean.json` del complemento GPU: TP, FP, FN y HD95 por imagen) y
> `analizar`. Tablas completas en `TABLAS_LOCAL.md`; datos en `RESUMEN_LOCAL.json` y los CSV de esta carpeta.
>
> **Pendiente:** la prueba contrafactual de atajos y la taxonomía exacta (`colab/Atajos_Errores_E2.ipynb`, GPU) y la
> revisión humana de 88 casos (`revision` y `medir-revision`). Ver al final.

## 1. Taxonomía de errores (fila 65)

Caso de error: par imagen-semilla con Dice < 0,80. Todas las redes superan los 100 casos: de 188 (TransUNet) a 482
(SegFormer). Como un mismo caso se repite en varias semillas, también se cuentan imágenes distintas: 63 a 125 con
el umbral de 0,80 y 128 a 186 con 0,90. Por eso los IC se calculan por grupo y no por caso.

**Forma del error (eje A).** La omisión total casi no ocurre: entre 0 y 3,6 %. El error se reparte entre dos formas:

- **Exceso lejos del rótulo** (A5, medido con HD95 como sustituto). Domina en las redes convolucionales pequeñas:
  FCN 0,50 [0,37; 0,63], U-Net 0,56 [0,43; 0,68] y SegNet 0,56 [0,43; 0,68].
- **Subsegmentación** (A3). Domina en SegFormer, con 0,55 [0,45; 0,65], y pesa en DeepLabv3+, U-Net++ y PIDNet
  (0,37 a 0,40).

TransUNet reparte sus errores entre las dos (0,33 y 0,34). Con el umbral de 0,90, A5 sigue primero en FCN, U-Net y
SegNet, y la subsegmentación pasa a ser la más frecuente en las demás redes, TransUNet incluida.

**Condición de la imagen (eje B), frente a su frecuencia en toda la validación:**

| Condición | En toda la validación | Entre los errores (rango de las 10 redes) |
|---|---:|---:|
| rótulo pequeño (Q1 de train) | 13 % | 15–31 % (top 3: 26–31 %, el doble) |
| varios rótulos | 44 % | 45–58 % |
| luz extrema | 8 % | 10–14 % |
| rótulo en el borde | 10 % | 0,4–13 % |
| sin condición | 22 % | 3–12 % |

Los errores se concentran en rótulos pequeños y en imágenes con varios rótulos. En las imágenes "limpias" casi no hay
errores. El ruido de etiqueta confirmado solo alcanza una imagen de validación, porque la revisión de ruido muestreó
sobre todo train.

**Errores compartidos.** 34 de 218 imágenes (0,156 [0,095; 0,224]) fallan en al menos 8 de las 10 redes, y 26 fallan
en todas. Por condición, 17 tienen varios rótulos, 10 un rótulo pequeño y 5 luz extrema. En el otro extremo, 88
imágenes no fallan en ninguna red. Una parte del error es dificultad propia de la imagen o de su anotación, no de la
arquitectura, como anticipa la guía.

**Cautela sobre A5.** El conjunto tiene rótulos reales sin máscara: el 10 % de las imágenes revisadas tenía un rótulo
cortado por el borde sin anotar (`resultados/ruido_etiqueta_e2/`). Parte del "exceso distante" puede ser un acierto
de la red contra una máscara incompleta. Para resolverlo, la hoja de revisión pregunta exactamente eso, y la corrida
de Colab mide las componentes predichas en lugar de usar HD95.

## 2. Subgrupos (fila 67)

Diferencia de cada subgrupo con el resto, con p de Holm dentro de cada red y atributo:

- **Tamaño.** El cuartil más pequeño pierde de 0,075 a 0,12 de Dice; es significativo en 6 de 10 redes, entre ellas
  las tres mejores. El efecto coincide con el de los estratos de área en test (`resultados/integracion_e2`).
- **Poca luz.** El tercil más oscuro pierde de 0,07 a 0,18. Es significativo en 7 de 10 redes; en TransUNet queda en
  p Holm 0,075. Es el subgrupo con la caída más grande en SegNeXt (−0,18) y SegFormer (−0,15).
- **Clase.** Organic-peroxide sale mejor en 9 de 10 redes (+0,11 a +0,17), pero son 11 imágenes de solo 6 grupos.
  En test, la clase "mixta" (varias clases en una imagen) pierde de 0,20 a 0,47 en todas las redes; en validación el
  signo es el mismo pero el efecto no es significativo tras Holm. Las demás clases tienen IC demasiado anchos (8 a 28
  imágenes) para afirmar una disparidad.
- **Borde.** Las imágenes con el rótulo en el borde salen mejor (+0,05 a +0,11). No es un efecto del borde: esos
  rótulos son primeros planos, con un área mediana de 0,063 frente a 0,027. El tamaño confunde la comparación.
- **Contraste y origen O1–O6.** No hay diferencias.

## 3. Atajos: lo que ya se puede decir (fila 13)

La guía pide comparar el desempeño con y sin la señal espuria identificada en el EDA. En el EDA, la señal candidata
era el **origen de captura** (carpeta del XML). Los seis orígenes principales (O1–O6, de 27 a 40 imágenes cada uno)
rinden igual. El origen menor, O7, sale mejor en las 10 redes en validación (+0,09 a +0,22, p Holm ≤ 0,035), pero son
5 imágenes (dos con varios rótulos y una con rótulo pequeño). En test (10 imágenes) el signo se repite en las 10 redes
(+0,04 a +0,12), aunque solo es significativo en dos (DeepLabv3+ y SegNeXt).

Lectura: el desempeño no depende del colector para casi todas las fotos. O7 es más fácil, y con tan pocas fotos no puede
separarse si es por cómo se capturó o por qué se fotografió.

Falta la prueba directa: quitar el contexto, borrar el rótulo y quitar el color (C1–C3).

## 4. Daño potencial (fila 67)

El error con más costo en la aplicación es no ver un rótulo de mercancía peligrosa. La **omisión grave** (recall
< 0,5) ocurre en el 3,8–7,7 % de los pares imagen-semilla en nueve redes y en el 20,3 % [14,3; 26,6] en SegFormer. El
test da cifras parecidas (3,0–9,5 % y 20,6 %).

En el top 3 la omisión grave se concentra con **poca luz**: 8–12 % en el tercil oscuro frente a 1–3 % en el resto. La
luz es la disparidad con consecuencias operativas: rótulos de noche, en bodegas o a contraluz se perderían más. Por
clase, las tasas más altas (oxygen, poison, infectious-substance: 0,13 a 0,20) tienen IC que llegan a 0, así que no
se puede afirmar una disparidad por clase con 8 a 28 imágenes.

No hay negativos reales en el conjunto. Las falsas alarmas se miden en negativos sintéticos (C2), pendiente en Colab.

## Qué falta y cómo se cierra

1. **Colab (GPU, necesita visto bueno).** Abrir `colab/Atajos_Errores_E2.ipynb`, correr las celdas 1 a 7 y bajar
   `atajos_errores_e2.zip` (unos 30 a 45 min en L4; el piloto lo estima antes). Descomprimirlo fuera del repositorio y
   correr `python scripts/errores_atajos_e2.py atajos --entrada CARPETA`. Escribe `RESUMEN_ATAJOS.json` y
   `TABLAS_ATAJOS.md`: decisión por red sobre contexto, color y rótulo borrado, falsas alarmas en negativos y
   taxonomía exacta.
2. **Revisión humana (unos 45 min).** Correr `python scripts/errores_atajos_e2.py revision --entrada CARPETA --voc
   RUTA/VOC2012 --destino CARPETA_FUERA_DEL_REPO`, que deja 22 páginas con 4 casos cada una y
   `hoja_revision_errores.csv`. Se marcan en la hoja el acuerdo con A y con B, y si el exceso es un rótulo sin anotar.
   Después, `python scripts/errores_atajos_e2.py medir-revision --hoja .../hoja_revision_errores.csv` escribe
   `REVISION_HUMANA.json`. Son 88 casos y no 90: con 30 por red no alcanzan las imágenes distintas en A1 y A2.
3. `python scripts/actualizar_checklist_errores.py` toma el estado de lo que exista: 13 y 67 pasan a C con
   `RESUMEN_ATAJOS.json`, y 65 con `REVISION_HUMANA.json`.
