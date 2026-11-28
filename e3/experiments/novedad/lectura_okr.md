# Lectura de texto completo: Optimal Kernel Representation (OKR)

> **Trazabilidad.** Leído el 2026-11-24 en IEEE Xplore por el proxy de la biblioteca de Uninorte (texto completo, §3.1 a
> §3.4, páginas 2 y 3 del PDF). Ma, H., Yang, C., Yuan, Y. y Wang, Q. (2023). *Optimal Kernel for Real-Time
> Arbitrary-Shaped Text Detection*. ICASSP 2023. doi 10.1109/ICASSP49357.2023.10094734.

## Qué hace

Detector de texto por segmentación de núcleos (backbone ResNet, FPN, mapa a 1/4). Su representación, OKR, tiene dos
pasos, y los dos son adaptativos a la geometría de cada instancia:

- **Erosión adaptativa** (Ec. 1): el tamaño de erosión es S = ⌈mín{len_k} / 2⌉, con len_k la longitud del lado k del
  polígono del texto y ⌈·⌉ el redondeo a impar. Es decir, la erosión depende del lado más corto (el grosor) de la
  instancia. La motivación es explícita: con un tamaño fijo, "los textos pequeños se pierden cuando el tamaño es
  grande, y los adyacentes no se separan cuando es pequeño".
- **Dilatación adaptativa** (Ec. 2): en texto cuadrilátero, ajusta el rectángulo rotado de área mínima a la región
  predicha y toma su lado más corto como tamaño de dilatación. En texto arbitrario, calcula el tamaño con una fórmula
  cerrada a partir del perímetro C y el área A de la región predicha: S = ⌈(C − √máx{C² − 16A, 0}) / 4⌉. En un
  rectángulo de lados a y b, C² − 16A = 4(a − b)², así que S = mín(a, b), el lado corto del núcleo. Como S es el
  tamaño del elemento estructurante (un diámetro), la distancia por lado es S/2: el núcleo de ancho w/2 que deja la
  erosión vuelve a ancho w. Es la inversa cerrada de su regla de erosión para rectángulos.
- Erosión y dilatación son **morfológicas** (núcleo de tamaño S), no desplazamientos de polígono.
- Pérdida: entropía cruzada ponderada del núcleo más una pérdida de centros de carácter virtuales (PEM, solo en
  entrenamiento). Sin mapa de umbral ni binarización diferenciable.

## Frente a UNet-AT

| Pieza de UNet-AT | OKR | Diferencia | Clase de diferencia (p. 12) |
|---|---|---|---|
| C1: D = φ(g)·A(1 − r²)/L, φ(g) = g²/(g² + g0²) | Erosión S = mín lado / 2, proporcional al grosor | OKR ya adapta la erosión al grosor de la instancia por la misma razón (no perder textos pequeños). C1 es otra regla: atenúa el desplazamiento de Vatti de DBNet de forma suave y solo cuando el grosor se acerca al paso del mapa, conserva la regla de DBNet en las regiones gruesas y cumple la condición de invertibilidad ρ' < 1 | Mecanismo (regla distinta), pero **la idea de erosionar según el grosor no es nueva** |
| C2a: d resuelto como punto fijo d = ρ(M(K', d)) | Dilatación con fórmula cerrada a partir de C y A, exacta para rectángulos | OKR ya calcula la dilatación a partir de la geometría del núcleo para invertir su erosión. C2a resuelve la inversa por punto fijo para cualquier regla con ρ' < 1 y cualquier polígono convexo, con la demostración de unicidad y de inversión exacta (Lema 2 y Teorema, `diseno.md` §3.5) | Mecanismo (procedimiento general y demostrado frente a fórmula cerrada para un caso) |
| C2b: expansión en inglete | Dilatación morfológica | Esquinas vivas frente a redondeadas o cuadradas según el elemento estructurante | Mecanismo (menor; Jiang et al. 2020 ya usan inglete) |
| Representación de DBNet (P, T, binarización diferenciable) | No la usa | — | No es aporte de UNet-AT; viene de DBNet |

## Consecuencia para R1

- **OKR pasa a ser el trabajo más cercano** y tiene que estar en la tabla comparativa y en la declaración de
  novedad, junto a ADNet y a Jiang et al. (2020).
- **La declaración no puede decir** que la erosión adaptativa al grosor o la dilatación calculada desde la geometría
  del núcleo son nuevas.
- **Lo que sí sostiene R1** (diferencia de mecanismo, no solo de aplicación):
  1. la inversión por punto fijo, general para cualquier regla de encogimiento contractiva y sin parámetros
     aprendidos;
  2. la demostración de inversión exacta en polígonos convexos, que OKR no da;
  3. la regla atenuada de C1 sobre el desplazamiento de Vatti, diseñada para cumplir esa condición;
  4. la aplicación a rótulos de mercancías peligrosas, que ningún trabajo hace (aplicación sola no basta, pero
     suma).
- **Riesgo residual:** un evaluador puede leer C1 + C2 como una variante de OKR. La defensa es el teorema y la
  generalidad del punto fijo. Conviene un brazo de inferencia con la dilatación de OKR (Ec. 2) sobre los mismos
  checkpoints, para medir la diferencia; queda como propuesta (ver `plan.md`).
