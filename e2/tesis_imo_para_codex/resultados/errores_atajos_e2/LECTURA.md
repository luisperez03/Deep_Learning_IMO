# Errores, subgrupos, daño potencial y atajos: lectura

> **Trazabilidad.** 22/11/2026. Protocolo escrito antes de calcular: `PROTOCOLO_ERRORES_ATAJOS_20261122.md` y
> `config/errores_atajos_e2_20261122.json` (SHA-256 en cada salida). Exploratorio: el test ya estaba observado; nada de
> esto cambia modelos, umbral ni checkpoints. Validación (218 imágenes, 122 grupos), 10 redes × 5 semillas; test solo
> como descripción secundaria. IC95 por bootstrap de grupos (4.000).
>
> **En local** (sin predicciones): `python scripts/errores_atajos_e2.py atributos --voc RUTA/VOC2012`, `extraer
> --diagnosticos RUTA/diag_descarga` (los `*_clean.json` del complemento GPU: TP, FP, FN y HD95 por imagen) y
> `analizar` → `TABLAS_LOCAL.md`, `RESUMEN_LOCAL.json` y los CSV.
>
> **En Colab** (L4, 50 de 50 corridas, unos 20 min): `colab/Atajos_Errores_E2.ipynb` sin cambios, con
> `scripts/atajos_gpu_e2.py` y la config embebidos (SHA-256 verificados). La salida está fuera del repositorio y en
> Drive (`tesis_imo_e2_atajos_v1`). Resumen local: `errores_atajos_e2.py atajos --entrada CARPETA` →
> `TABLAS_ATAJOS.md` y `RESUMEN_ATAJOS.json`.
>
> **Revisión humana** de 88 casos (fila 65), por confirmación global del revisor: sección 6.

## 1. Taxonomía de errores (fila 65)

Caso de error: par imagen-semilla con Dice < 0,80. Todas las redes superan los 100 casos: de 188 (TransUNet) a 482
(SegFormer). Como un mismo caso se repite en varias semillas, también se cuentan imágenes distintas: 63 a 125 con
el umbral de 0,80 y 128 a 186 con 0,90. Por eso los IC se calculan por grupo y no por caso.

**Forma del error (eje A, exacta con las componentes predichas en Colab).** La omisión total casi no ocurre: entre
0 y 3,6 %. Tres formas reúnen casi todo el error, y cada red tiene la suya:

- **Subsegmentación** (se queda corta). Es la forma principal de SegFormer, con 0,55 [0,45; 0,65], y de DeepLabv3+,
  U-Net++ y PIDNet (0,37 a 0,40).
- **Exceso pegado al rótulo** (se pasa del borde). Es la principal de U-Net, VM-UNet, SegNet y SegNeXt (0,37 a
  0,39).
- **Exceso lejos del rótulo.** Pesa sobre todo en FCN, U-Net y SegNet (0,28 a 0,30) y casi nada en SegFormer (0,05).

TransUNet reparte sus errores entre las dos primeras (0,33 y 0,33). El sustituto HD95 que se usó en local exageraba
el exceso lejano: coincide con la forma exacta en el 72–94 % de los casos.

Las redes preentrenadas grandes pierden **rótulos enteros** en imágenes con varios: de 78 a 159 veces en DeepLabv3+,
PIDNet, U-Net++ y SegFormer, frente a 1 o 2 en FCN, U-Net y SegNet. Es la suma de las 5 semillas sobre las imágenes de
validación con más de un rótulo.

**Condición de la imagen (eje B), frente a su frecuencia en toda la validación:**

| Condición | En toda la validación | Entre los errores (rango de las 10 redes) |
|---|---:|---:|
| rótulo pequeño (Q1 de train) | 13 % | 15–31 % (top 3: 26–31 %, el doble) |
| varios rótulos | 44 % | 45–58 % |
| luz extrema | 8 % | 10–14 % |
| rótulo en el borde | 10 % | 0,4–13 % |
| sin condición | 22 % | 3–12 % |

Los errores se concentran en rótulos pequeños y en imágenes con varios rótulos. En las imágenes sin ninguna condición
difícil casi no hay errores. El ruido de etiqueta confirmado solo alcanza una imagen de validación, porque la revisión
de ruido muestreó sobre todo train.

**Errores compartidos.** 34 de 218 imágenes (0,156 [0,095; 0,224]) fallan en al menos 8 de las 10 redes, y 26 fallan
en todas. Por condición, 17 tienen varios rótulos, 10 un rótulo pequeño y 5 luz extrema. En el otro extremo, 88
imágenes no fallan en ninguna red. Una parte del error es dificultad propia de la imagen o de su anotación, no de la
arquitectura, como anticipa la guía.

**Cautela sobre el exceso lejano.** El conjunto tiene rótulos reales sin máscara: el 10 % de las imágenes revisadas
tenía un rótulo cortado por el borde sin anotar (`resultados/ruido_etiqueta_e2/`). Parte del exceso lejano puede ser
un acierto de la red contra una máscara incompleta. En los paneles aparecen también objetos parecidos, como letreros
verdes de salida. La hoja de revisión pregunta cuál de los dos casos es.

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

## 3. Atajos (fila 13)

**Por subconjuntos con la señal del EDA.** En el EDA, la señal candidata era el **origen de captura** (carpeta del
XML). Los seis orígenes principales (O1–O6, de 27 a 40 imágenes cada uno) rinden igual. El origen menor, O7, sale
mejor en las 10 redes en validación (+0,09 a +0,22, p Holm ≤ 0,035), pero son solo 5 imágenes, dos con varios rótulos
y una con un rótulo pequeño. En test (10 imágenes) el signo se repite en las 10 redes (+0,04 a +0,12), aunque solo
es significativo en dos (DeepLabv3+ y SegNeXt). El desempeño no depende del colector para casi todas las fotos.

**Prueba contrafactual** (Δ = Dice con la imagen cambiada − Dice limpio; reglas fijadas antes, δ = 0,02, Holm sobre
20 pruebas):

| Red | Sin contexto Δ [IC95] | Decisión | Gris Δ [IC95] | Rótulo borrado: zona aún marcada [IC95] |
|---|---:|---|---:|---:|
| TransUNet | −0,117 [−0,144; −0,091] | dependencia | −0,332 [−0,382; −0,284] | 0,036 [0,022; 0,052] |
| FCN | +0,025 [0,012; 0,040] | sin dependencia | −0,402 [−0,453; −0,352] | 0,014 [0,007; 0,023] |
| U-Net | +0,022 [0,009; 0,037] | sin dependencia | −0,331 [−0,381; −0,284] | 0,014 [0,007; 0,024] |
| VM-UNet | −0,022 [−0,040; −0,004] | indeterminado | −0,288 [−0,325; −0,251] | 0,055 [0,033; 0,081] |
| SegNet | +0,036 [0,021; 0,051] | sin dependencia | −0,387 [−0,441; −0,334] | 0,015 [0,008; 0,025] |
| DeepLabv3+ | +0,002 [−0,014; 0,019] | sin dependencia | −0,368 [−0,418; −0,317] | 0,069 [0,045; 0,096] |
| U-Net++ | 0,000 [−0,019; 0,020] | sin dependencia | −0,300 [−0,341; −0,258] | 0,082 [0,058; 0,109] |
| PIDNet-S | +0,001 [−0,019; 0,023] | sin dependencia | −0,324 [−0,367; −0,281] | 0,076 [0,053; 0,103] |
| SegNeXt-T | −0,050 [−0,080; −0,020] | indeterminado | −0,313 [−0,359; −0,268] | 0,073 [0,049; 0,101] |
| SegFormer-B0 | −0,215 [−0,263; −0,168] | dependencia | −0,419 [−0,469; −0,370] | 0,095 [0,065; 0,128] |

Lectura:

- **Contexto.** Seis redes segmentan el rótulo por el rótulo mismo. FCN, U-Net y SegNet incluso mejoran sin contexto:
  el fondo les genera falsos positivos. Las dos con atención global dependen del contexto: TransUNet pierde 0,12 y
  SegFormer 0,22. VM-UNet y SegNeXt quedan en indeterminado: caen, pero sin superar con claridad el margen δ.
  - Cautela: un fondo plano es una imagen fuera de distribución, y a las redes con atención global les cambia todo el
    mapa de atención. La caída mide la dependencia del contexto o la sensibilidad a ese artefacto, sin poder
    separarlas.
- **Rótulo borrado.** Ninguna red sigue marcando el rótulo donde ya no está: entre el 1,4 y el 9,5 % de la zona, con
  todas las cotas inferiores por debajo del 0,10 fijado. No hay atajo del tipo "aquí suele haber un rótulo".
- **Color.** Las 10 redes caen de 0,29 a 0,42 sin color. Dependen mucho del color, pero en un rótulo de mercancía
  peligrosa el color es parte del significado (rojo inflamable, amarillo comburente). Se informa como dependencia, no
  como atajo. La consecuencia práctica es que una cámara en blanco y negro o una luz que altere el color degradaría
  todo el benchmark.

## 4. Daño potencial (fila 67)

El error con más costo en la aplicación es no ver un rótulo de mercancía peligrosa. La **omisión grave** (recall
< 0,5) ocurre en el 3,8–7,7 % de los pares imagen-semilla en nueve redes y en el 20,3 % [14,3; 26,6] en SegFormer. El
test da cifras parecidas (3,0–9,5 % y 20,6 %).

En el top 3 la omisión grave se concentra con **poca luz**: 8–12 % en el tercil oscuro frente a 1–3 % en el resto. La
luz es la disparidad con consecuencias operativas: rótulos de noche, en bodegas o a contraluz se perderían más. Por
clase, las tasas más altas (oxygen, poison, infectious-substance: 0,13 a 0,20) tienen IC que llegan a 0, así que no
se puede afirmar una disparidad por clase con 8 a 28 imágenes.

**Falsas alarmas.** No hay imágenes sin rótulo en el conjunto. Se midieron en **negativos sintéticos**: la misma foto
con los rótulos anotados borrados. La red marca al menos 50 píxeles en el 31–59 % de ellos: FCN, U-Net y SegNet,
0,31–0,33; las demás, 0,44–0,59.

- La mayor parte viene de lo que queda en la foto. Cuando la imagen limpia ya tenía exceso lejano, la alarma aparece
  en el 70–95 % de los casos: rótulos sin anotar u objetos parecidos.
- Cuando no lo tenía, baja al 9–11 % en FCN, U-Net y SegNet, y queda en 33–46 % en el resto.
- Para un uso en campo, la tasa de falsas alarmas es tan relevante como la de omisión. No hay negativos reales ni
  validación en operación; queda como limitación.

## 5. Control de la corrida

La imagen limpia reproduce exactamente el Dice por imagen de los diagnósticos GPU en 9 redes: diferencia máxima 0 en
1.090 pares cada una. **SegNeXt-T no**: 792 de 1.090 pares difieren, hasta 0,13. Su decodificador (factorización de
matrices) se inicializa al azar en cada pasada. Las cifras de SegNeXt traen ese ruido adicional, que no sesga Δ pero
lo hace menos preciso. El Dice medio de validación sí coincide (0,759).

## 6. Revisión humana (fila 65)

**Material.** `python scripts/errores_atajos_e2.py revision --entrada CARPETA --voc RUTA/VOC2012 --destino
CARPETA_FUERA_DEL_REPO` dejó `REVISION_ERRORES.pdf` (22 páginas, 88 casos del top 3: 29 de TransUNet, 29 de U-Net y
30 de FCN) y `hoja_revision_errores.xlsx`. La forma que muestra la hoja es la exacta de Colab. Son 88 casos y no 90:
con 30 por red no alcanzan las imágenes distintas en A1 y A2.

**Cómo se llenó (22/11).** El revisor miró los 88 casos en el PDF y declaró que todas las clasificaciones automáticas
eran correctas, pero no marcó la hoja. Con su autorización expresa, la hoja se llenó por script:

- "sí" en el acuerdo con la forma y con la condición de los 88 casos (`registrar_confirmacion_global.py`);
- en los 35 casos de forma A5 o A6, "no" a "el rojo lejano es un rótulo sin anotar", según su respuesta global de
  que siempre era un error de la red; "no aplica" en los 53 restantes (`registrar_columna_i.py`).

Los dos scripts están en la carpeta de trabajo del estudio (`ejemplos/2026-11-22_revision_errores_e2/`). La hoja
vacía se conserva aparte. `REVISION_HUMANA.json` guarda el SHA-256 de la hoja llena.

**Alcance de la revisión.** El revisor expresó conformidad global tras examinar los 88 paneles. La hoja fue completada por script a partir de esa respuesta; no contiene decisiones registradas individualmente por el revisor. Por ello, los conteos e intervalos calculados desde esa hoja no se interpretan como una estimación de acuerdo humano independiente y se omiten de esta síntesis.

La revisión aporta una comprobación cualitativa: el revisor no señaló categorías incorrectas. No permite cuantificar la frecuencia de errores de clasificación ni concluir, caso por caso, que el exceso lejano corresponde siempre a un error de la red. Además, las categorías automáticas eran visibles, por lo que no fue una evaluación ciega.

Para completar esta validación se necesita registrar la decisión del revisor para cada caso, incluidos los desacuerdos y los casos dudosos. La evidencia original se conserva para trazabilidad. La fila 65 del checklist queda parcial hasta contar con ese registro.
