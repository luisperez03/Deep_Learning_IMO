# Protocolo experimental del Entregable 3

> **Trazabilidad.** Escrito el 2026-11-23. Se congela con la etiqueta de Git `e3-protocolo` antes de entrenar el
> modelo original y antes de las semillas 6 a 10 de los rivales. Desde la etiqueta no cambian la arquitectura, la
> pérdida, las reglas C1 y C2, g0, AJUSTE_PX, la receta, las semillas, la familia, las pruebas ni los criterios.
> Cualquier cambio reinicia el protocolo con otra etiqueta y se reporta como nueva versión (p. 18).
>
> **Versión 2 (2026-11-24, etiqueta `e3-protocolo-v2`), antes de cualquier corrida del E3.** Cambios: el presupuesto de
> ajuste (§2), que la v1 describía mal; SegFormer-B0 con su configuración ganadora; la evidencia de la limitación
> (`estandar.md` §3); y la publicación de los pesos (§11). Nada se había corrido con la v1.
>
> **Base:** `entregable_3/diseno.md` (§3 y §5), adaptado al benchmark entregado en el E2 (10 redes desde cero,
> 5 semillas) según `DECISION_E3_2026-11-21.md` y las decisiones de `plan.md`. El estándar está en `estandar.md`.

## 1. El modelo original: UNet-AT

- **Esqueleto:** la U-Net compacta del E2 (`modelos_base.SmallUNet`: 3 niveles de 16, 32 y 64 canales, sin
  BatchNorm), desde cero, sin pesos preentrenados, igual que los rivales.
- **Representación de DBNet** (Liao et al., 2020), transferida desde la detección de texto en escena: cabeza de
  probabilidad del núcleo P, cabeza de umbral T y binarización aproximada B = σ(k(P − T)), k = 50. Pérdida: la de
  DBNet según el código del E2 propio (`src/original_model.py` documenta los pesos).
- **C1, encogimiento atenuado por grosor:** D = φ(g)·A(1 − r²)/L, φ(g) = g²/(g² + g0²), r = 0,4, g0 = 4 px (paso
  del mapa más profundo de la U-Net compacta, 1/4 de 256; fijado por la arquitectura).
- **C2, expansión consistente:** d tal que d = regla(expandir(K', d)), por bisección (16 pasos) y refinamiento en
  píxeles, con expansión en inglete propia (límite 2) y AJUSTE_PX = 0,75 (calibrado en 200 imágenes de train,
  `design_log.md`).
- Inferencia: binarizar P con 0,3; descartar componentes con P media < 0,5 o área < 9 px; expandir con C2.
- Código: `src/geometria_at.py`, `src/original_model.py`, `training/entrenar_original.py`. Su SHA-256 queda en
  `procedencia.md` al congelar.

**Brazos entrenados (dos):**

| Brazo | Etiquetas | Papel |
|---|---|---|
| `dbnet` | Regla de DBNet (φ ≡ 1) | Base del mecanismo: la misma red con la representación de DBNet sin cambios |
| `dbnet_at` | C1 | El modelo original |

Las otras 6 variantes de la ablación factorial 2³ (C1, C2a, C2b) salen de esos checkpoints cambiando solo la
inferencia (`diseno.md` §5.1).

## 2. Lo que se hereda del E2 y no se toca

- **Datos:** la partición del E2 (1.015 train, 218 validación con 122 grupos, 218 test con 122 grupos), los mismos
  archivos y SHA-256 (`resultados/preentrenamiento/freeze_manifest.json` del repo del E2), letterbox a 256 × 256.
- **Receta:** la de la U-Net compacta en `config/experimento_e2.json`: Adam, lr 1e-3, wd 0, lote 2, hasta 30
  épocas, paciencia 5, mismo orden de lotes por semilla y época, selección del checkpoint por Dice medio por grupo en
  validación. En el original, ese Dice se mide después del posproceso C2.
- **Presupuesto de ajuste (corregido en la v2):** el cierre del E2 corrió para las 10 redes una búsqueda equitativa
  de 4 intentos (receta × 10^u en lr y weight decay en {0; 1e-4; 1e-2}, 1 semilla, 10 épocas; fila 37,
  `resultados/complemento_entreno_e2/RESUMEN.md`). La versión 1 de este protocolo decía, por error, que no hubo
  búsqueda. Regla v2:
  - **Rivales:** la receta del benchmark, salvo **SegFormer-B0**, que corre con la configuración ganadora (t = 2:
    AdamW, lr 5,57e-4, wd 1e-4) porque la receta lo dejaba mal ajustado (0,660 frente a 0,782 de Dice en
    validación) y el enunciado prohíbe comparar contra rivales mal ajustados. Se corren sus 10 semillas con esa
    configuración y se registra como corrección del bloque A. En las otras 9 la búsqueda no mueve el Dice más de δ.
  - **Original:** la receta de su esqueleto como configuración confirmatoria, igual que los rivales, y la misma
    búsqueda de 4 intentos con el mismo presupuesto, exploratoria (`e3/config/hpo_original_t*.json`, generadas por
    `training/configs_e3.py` con la secuencia del E2).
- Si la búsqueda de arquitectura de las
  compactas del E2 (checklist 38, B14) elige otra U-Net compacta, la U-Net rival se corre con la elegida y el
  original se mantiene sobre la del E2. Se declara la asimetría de esfuerzo de diseño (Nd = 5, `design_log.md`).
- **Rivales:** sus checkpoints y resultados del E2 quedan congelados con hash al cerrar el bloque A. No se reajustan.

## 3. Semillas

- **R = 10 para todos los modelos de la familia y para los dos brazos:** 11, 23, 37, 53, 71 (las del E2) y 89, 97,
  101, 103, 107 (las cinco siguientes de la lista fijada en el E2, `entregable_2/codigo/fases.py`).
- **Exclusión:** una corrida se excluye solo si termina con pérdida no finita o sin checkpoint válido. Se reemplaza
  por la primera semilla libre de la reserva (127, 131, 137, 139, 149) en todos los modelos a la vez, para mantener
  el pareo. Se reporta.
- Por qué 10 y no más: ver §7.

## 4. Métricas

- **Primaria:** Dice por grupo en test. El valor de un grupo es la media de sus imágenes en cada semilla, y luego la
  media sobre las 10 semillas.
- **Tercil chico:** el tercio de los 122 grupos de test con menor área etiquetada media, calculado con las
  etiquetas y sin ningún modelo; el mismo para todos.
- **Secundarias preregistradas:** IoU y HD95 por grupo; Dice en los terciles 2 y 3; brecha "tercil chico − resto";
  Dice por región por intervalos de área (174, 390, 719 y 1.398 px).
- **Adecuación:** supera a la máscara vacía y a la mayoritaria; brecha entre train y validación del checkpoint
  elegido; consistencia entre validación y test.

## 5. Familia confirmatoria principal (m = 10)

**Rivales:** FCN, SegNet y U-Net compactas, DeepLabv3+ R18, UNet++ R18, PIDNet-S, SegNeXt-T, SegFormer-B0,
TransUNet y VM-UNet. **Referente fuera de la familia:** el baseline clásico del E2 (Dice 0,645), que no es un
baseline fuerte frente a redes que pasan de 0,80. Se reporta en tablas y en el diagrama de diferencia crítica sin
prueba confirmatoria. La lista queda cerrada aquí.

**Contraste j:** dⱼ = Dice(UNet-AT) − Dice(rival j), por grupo, en los 122 grupos de test.

**Tres pruebas por contraste**, cada una con Holm sobre los 10 contrastes:
1. Wilcoxon de rangos con signo, pareado por grupo, bilateral.
2. Bootstrap pareado por grupos: 10.000 réplicas; p bilateral por inversión del intervalo.
3. t pareado por grupo, con Shapiro-Wilk sobre dⱼ reportado.

**UNet-AT supera al rival j** si se cumplen las cuatro condiciones: el mayor de los tres p ajustados es menor que
0,05; la diferencia media es al menos δ = 0,02; |δ de Cliff| ≥ 0,147; el límite inferior del IC 95 % bootstrap es
mayor que 0.

**Complementos que no deciden:** Friedman con Iman-Davenport (grupos como bloques), Nemenyi y diagrama de diferencia
crítica con el original incluido; modelo de efectos mixtos y = µ + αⱼ + u_grupo + v_semilla + ε sobre el Dice por
imagen; probabilidad de superioridad P(original > rival) por grupos.

**Sensibilidad de la conclusión:** otra prueba (t sola), otra métrica (IoU), otros subconjuntos de semillas (las 5
primeras y las 5 últimas).

## 6. Hipótesis del mecanismo (familia aparte)

**H.** C1 hace que el núcleo de los rótulos pequeños no sea más delgado que el paso del mapa más profundo, y C2 evita
que eso cueste borde. Por tanto, UNet-AT (`dbnet_at`) sube el Dice por grupo de su base (`dbnet`) en el tercil chico
de test en al menos δ = 0,02, sin bajarlo más de δ en los 122 grupos.

- **H1a, superioridad en el tercil chico:** las tres pruebas del §5 a α = 0,05; decide el p más alto; IC 95 % por
  encima de 0 y diferencia de al menos δ.
- **H1b, no inferioridad en todos los grupos:** límite inferior del IC 90 % bootstrap mayor que −δ.
- **Predicciones secundarias**, que se reportan cumplidas o fallidas:
  - casi toda la ganancia en el tercil chico viene de C1 y no de C2 (la variante `c2` sin C1 no la consigue);
  - el HD95 del tercil chico baja;
  - la ventaja desaparece al quitar C1 y no aparece con C1 sin C2a;
  - UNet-AT frente a la U-Net compacta rival: la brecha "tercil chico − resto" se reduce.
- **Se refuta** si no se cumple H1a o no se cumple H1b.

## 7. Potencia y número de semillas

`experiments/potencia_e3.py` → `potencia_e3.json`, con las 5 semillas del E2 en validación y la U-Net compacta como
esqueleto del original:

- R necesario para δ = 0,02, potencia 0,90 y Holm (z = 2,807): de 10 (DeepLabv3+) a 48 (VM-UNet); con la cota
  superior del IC 80 % de σ̂_d, de 37 a 178.
- Con R = 10, la diferencia mínima detectable va de 0,020 a 0,043, y la potencia conjunta aproximada es 0,001.
- **Decisión declarada:** R = 10. Subir a 48 semillas cuesta unas 125 h de L4 solo en los rivales y no cabe en el
  plazo. Por eso R3 se declara no alcanzable en `estandar.md`, y cada contraste se reporta con su diferencia mínima
  detectable. Un resultado no significativo se lee como ausencia de evidencia, no como evidencia de ausencia.

## 8. Ablación (familia aparte, p. 20)

- Las 8 variantes del factorial 2³ de C1, C2a y C2b (`diseno.md` §5.1), con los checkpoints de los dos brazos.
- Cada variante contra el modelo completo, con las tres pruebas y Holm dentro de la familia (m = 7), en todos los
  grupos y en el tercil chico (dos familias de 7).
- Efectos principales e interacciones con IC 95 % bootstrap por grupos.
- Control de capacidad: no aplica, porque ninguna variante cambia parámetros ni FLOPs.
- **Comparación con el trabajo más cercano** (agregada el 2026-11-24, antes de congelar y sin evaluar nada):
  `okr_sobre_at` usa los núcleos del brazo `dbnet_at` y la dilatación de Optimal Kernel (Ma et al., 2023, Ec. 2,
  `src/geometria_at.distancia_okr`) en lugar de C2. Un contraste, `dbnet_at` − `okr_sobre_at`, con las tres pruebas,
  en todos los grupos y en el tercil chico, a α = 0,05 sin familia (es exploratorio y no decide R3). Mide si el punto
  fijo de C2 rinde más que la fórmula cerrada de OKR con los mismos núcleos. La erosión de OKR no se entrena (sería
  otro brazo en GPU): se declara.

## 9. Disciplina de diseño y test

- **Partición de desarrollo:** lo que tocó el diseño fue train (calibración y oráculo), 6 imágenes de validación y
  el oráculo de validación sin modelo (`design_log.md`, Nd = 5). Ningún dato de test.
- **Partición de confirmación:** test de HAZMAT-13.
- **Test ya observado:** los rivales se evaluaron en test en el E1 y en el E2. El original no se ha evaluado nunca
  en test. La evaluación del E3 en test es una lectura más de un conjunto ya visto para los rivales, y la primera
  para el original. Se declara en el informe.
- El test del original se evalúa una sola vez, después de terminar las 10 semillas de los dos brazos y de congelar
  los resultados de los rivales.

## 10. Robustez, eficiencia y calidad (Cuadro 5)

| Estudio | Qué se hace | Qué no |
|---|---|---|
| Estabilidad por semilla | Cajas, IC, coeficiente de variación y Kruskal-Wallis entre semillas, para todos | — |
| Perturbaciones | Ruido, desenfoque, brillo y oclusión en 3 niveles, por inferencia, para todos | — |
| Eficiencia y Pareto | Parámetros, FLOPs, latencia en CPU (ONNX Runtime) y en GPU, memoria pico, horas de entrenamiento | Energía: solo estimada (horas × TDP) |
| Calibración | Rivales: ECE, Brier, pendiente e intercepto, antes y después de temperature scaling ajustado en validación | El original no da probabilidad por píxel de la región completa (P es del núcleo). Se reporta la calibración de P frente al núcleo y se declara la limitación |
| Sensibilidad a hiperparámetros propios | g0 en {2, 8} px, 3 semillas, en validación, exploratorio | — |
| Curva de aprendizaje | Solo el original y su base, 25 % y 50 % de los grupos de train, 3 semillas | Los rivales no se recorren: se declara |
| Cambio de distribución | — | Sin conjunto externo (ver `estandar.md`) |
| Análisis de errores | Por área de la región y número de rótulos, con los errores compartidos con los rivales | — |
| Interpretabilidad | Grad-CAM y prueba de borrado del original frente al mejor rival, estabilidad entre semillas | — |

## 11. Publicación de los pesos (v2)

El E2 declaró que no redistribuye imágenes ni pesos: la licencia MIT del repositorio de HAZMAT-13 solo cubre el código
(fila 08 del cierre del E2). Por eso los artefactos ONNX no van al repositorio público. El dashboard los descarga al
arrancar desde un almacén privado con un token guardado como secreto de Render, y solo devuelve predicciones. La galería
no muestra imágenes del conjunto; la predicción en vivo usa las imágenes que sube quien usa la app, sin guardarlas.
