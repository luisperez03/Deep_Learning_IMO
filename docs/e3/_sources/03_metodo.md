(sec-metodo)=
# Método: UNet-AT

## Formulación

El esqueleto es la U-Net compacta del E2 (anchos 16, 32 y 64, sin BatchNorm, desde cero). Dos cabezas de convolución
1×1 producen la probabilidad del núcleo $P$ y el mapa de umbral $T$. La binarización aproximada es

$$
B = \sigma\big(k\,(P - T)\big), \qquad k = 50
$$ (eq-binarizacion)

y la pérdida total es

$$
\mathcal{L} = 5\,\mathcal{L}_s + 10\,\mathcal{L}_t + \mathcal{L}_b
$$ (eq-perdida)

con $\mathcal{L}_s$ la entropía cruzada binaria con minería de negativos difíciles 1:3 sobre $P$, $\mathcal{L}_t$ la
pérdida L1 de $T$ dentro de la banda y $\mathcal{L}_b$ la pérdida Dice de $B$ (según {cite:p}`liao2020dbnet`; pesos
confirmados en `MODELO_ORIGINAL_ESTADO.md`).

**C1, encogimiento atenuado por grosor.** Para una región de área $A$, perímetro $L$ y radio inscrito $g$, el
desplazamiento del borde al núcleo es

$$
D = \rho(K) = \varphi(g)\,\frac{A\,(1 - r^{2})}{L}, \qquad
\varphi(g) = \frac{g^{2}}{g^{2} + g_0^{2}}, \quad r = 0{,}4,\; g_0 = 4\ \text{px}
$$ (eq-c1)

La regla de DBNet es el caso $\varphi \equiv 1$.

**C2, expansión consistente.** En inferencia, la distancia de expansión $d$ es el punto fijo de

$$
d = \rho\big(M(K', d)\big)
$$ (eq-c2)

donde $K'$ es el núcleo predicho y $M(K', d)$ su desplazamiento en inglete (límite 2). Se resuelve por bisección
(16 pasos) y refinamiento en píxeles, con un ajuste de 0,75 px calibrado en 200 imágenes de entrenamiento
(`design_log.md`).

## Pseudocódigo de la inferencia

```{list-table} Inferencia de UNet-AT (C2)
:header-rows: 1
:name: tab-pseudocodigo

* - Elemento
  - Contenido
* - Entradas
  - Imagen $I$ (3×256×256, letterbox); pesos entrenados; umbral 0,3; parámetros $r$, $g_0$, ajuste.
* - Salidas
  - Máscara binaria 256×256 del rótulo.
* - Paso 1
  - Calcular $P$ y $T$ con la red; binarizar $P$ con 0,3.
* - Paso 2
  - Extraer componentes conexos; descartar los de $P$ media menor que 0,5 o área menor que 9 px.
* - Paso 3
  - Por cada componente $K'$: buscar $d$ con la ecuación {eq}`eq-c2` por bisección en $[0, d_{\max}]$.
* - Paso 4
  - Refinar $d$ en píxeles (candidatos vecinos) y expandir $K'$ con inglete propio.
* - Paso 5
  - Unir las regiones expandidas en la máscara de salida.
* - Criterio de parada
  - Bisección: 16 iteraciones. Refinamiento: ningún candidato vecino mejora la consistencia
    $\lvert d - \rho(M(K', d))\rvert$.
```

Código: `src/geometria_at.py`, `src/original_model.py`, `training/entrenar_original.py` (SHA-256 en
`experiments/procedencia.md`).

(sec-teorema)=
## Teorema de inversión exacta

Texto fuente: `entregable_3/diseno.md`, §3.5. Se conserva el argumento recuperado para revisión.
La comprobación numérica de la implementación no valida todos sus pasos formales; la revisión
independiente de esta demostración permanece pendiente. Su alcance continuo tampoco se extiende
automáticamente al posproceso discreto ni a núcleos predichos con error.

**Notación.** $K=\bigcap_i\{x:\ n_i\!\cdot\! x\le b_i\}$ es un polígono convexo con normales exteriores unitarias
$n_i$ y ángulos interiores $\theta_i$; $K_{-D}=\{x\in K:\ \operatorname{dist}(x,\partial K)\ge D\}$; el desplazamiento
en inglete de $P=\bigcap_j\{n_j\!\cdot\! x\le c_j\}$ es $M(P,d)=\bigcap_j\{n_j\!\cdot\! x\le c_j+d\}$.

**Supuestos.**

- S1. Ningún lado colapsa: $D<D^{*}(K)$, la menor distancia a la que un lado de $K_{-D}$ se reduce a un punto. En
  un rectángulo de ancho $w$, $D^{*}=w/2$ y la regla da $D\le 0{,}42\,(w/2)$.
- S2. Ángulos interiores $\theta_i\ge 60^\circ$, para que el límite de inglete 2 no recorte esquinas.

**Lema 1 (el núcleo es el polígono de rectas desplazadas).** Para $x\in K$,

$$
\operatorname{dist}(x,\partial K)=\min_i\,(b_i-n_i\!\cdot\! x)
$$ (eq-lema1)

y por tanto $K_{-D}=\bigcap_i\{n_i\!\cdot\! x\le b_i-D\}$.

*Demostración.* $\partial K\subset\bigcup_i\ell_i$, así que $\operatorname{dist}(x,\partial K)\ge\min_i
\operatorname{dist}(x,\ell_i)$. Recíprocamente, con $i^{*}$ el índice del mínimo, el rayo $x+t\,n_{i^{*}}$ sale de $K$
en algún $t\le b_{i^{*}}-n_{i^{*}}\!\cdot\! x$, de modo que $\operatorname{dist}(x,\partial K)\le b_{i^{*}}-
n_{i^{*}}\!\cdot\! x$. Las dos desigualdades dan la igualdad. $\square$

**Lema 2 (unicidad del punto fijo).** Bajo S1, con $K_1=K_{-D}$ y $h(d)=\rho(M(K_1,d))-d$, la función $h$ es
estrictamente decreciente en $d\ge 0$.

*Demostración.* Con $A_1$, $L_1$ y $g_1$ el área, el perímetro y el radio inscrito de $K_1$, las fórmulas de Steiner
del desplazamiento en inglete dan

$$
A(d)=A_1+L_1d+c\,d^{2},\qquad L(d)=L_1+2c\,d,\qquad g(d)=g_1+d,\qquad c=\sum_i\cot(\theta_i/2)>0
$$ (eq-steiner)

(lo último por el Lema 1). Con $q=A/L$ se tiene $q'(d)=1-2cA/L^{2}\le 1$. Por la desigualdad de Bonnesen
{cite:p}`osserman1979`, $gL\ge A+\pi g^{2}$, de modo que $q\le g$. Entonces
$\rho'(d)\le(1-r^{2})\,(g\varphi'(g)+\varphi(g))=(1-r^{2})\,(g\varphi)'$. Para $\varphi=g^{2}/(g^{2}+g_0^{2})$ el máximo
de $(g\varphi)'$ es $9/8$, en $g=\sqrt3\,g_0$; luego $\rho'\le 0{,}84\cdot 1{,}125=0{,}945<1$ y $h'=\rho'-1<0$. Con
$\varphi\equiv1$, $\rho'\le 0{,}84$. $\square$

**Teorema (C2 invierte a C1 y a la regla de DBNet).** Sea $K$ convexo con S1 y S2, $D=\rho(K)$ y $K_1=K_{-D}$. La
ecuación {eq}`eq-c2` tiene solución única $d=D$ y $M(K_1,D)=K$.

*Demostración.* Por el Lema 1, $K_1=\bigcap_i\{n_i\!\cdot\! x\le b_i-D\}$ y, por S1, los $m$ semiplanos siguen siendo
lados de $K_1$. Entonces $M(K_1,D)=\bigcap_i\{n_i\!\cdot\! x\le b_i\}=K$, y por S2 el límite de inglete no recorta esa
intersección. Así $\rho(M(K_1,D))=\rho(K)=D$: $D$ es raíz y, por el Lema 2, la única. $\square$

**Corolario (la expansión fija no puede).** Con $r'$ fijo, DBNet expande con $D'=A_1r'/L_1$ y bordes redondeados; no
hay una $r'$ que reconstruya a la vez un cuadrado y un rectángulo muy alargado, y el redondeo pierde $(4-\pi)D'^{2}$
por rectángulo aun con la $r'$ correcta.

**Qué violan los datos.** La discretización en píxeles (la inversa deja de ser única en ±1 px), las regiones reales
no convexas o con perspectiva, y los errores de la red (que el teorema no corrige). Por eso es motivación, no
garantía.

```{admonition} Pendiente
:class: warning
Verificación local con pytest (27/11/2026): 11 pruebas aprobadas y una omitida con OpenCV 4.11.0.
La discrepancia inicial del rombo de 10 px se reprodujo variando solo OpenCV: la versión 4.10
rasteriza algunas posiciones subpíxel de forma distinta y produce Dice 0,9730335; 4.11 produce
0,9822501, consistente con el 0,9823 histórico. Las 21 comparaciones ampliadas pasan con 4.11.
No se cambió la tolerancia ni el archivo de geometría. La versión queda fijada en el entrenamiento
y registrada en su identidad para impedir mezclar checkpoints entre estos entornos.
La equivalencia directa con `entregable_2/codigo/etiquetas.py` sigue omitida porque falta ese archivo.
Las pruebas numéricas no sustituyen una demostración formal ni la evaluación de un modelo entrenado completo.
```

## Complejidad

Con $H\times W$ píxeles, $n$ regiones por imagen y una región de área $a$ con caja recortada de área
$a_c\approx(\sqrt a+2m)^2$:

- **Red:** $O(HW)$ por imagen, con 17 parámetros más que la U-Net compacta (cabezas 1×1).
- **Etiquetas (C1):** una transformada de distancia por región, $O(nHW)$, con constante algo mayor que DBNet.
- **Inferencia (C2):** por región, unas 32 evaluaciones (16 de bisección y unas 16 del refinamiento), cada una
  $O(v+a_c)$ con $v$ vértices; total $O(32\,n\,a_c)$ contra $O(n\,a)$ del posproceso de DBNet.

Medido en CPU (`MODELO_ORIGINAL_ESTADO.md`): unos 0,3 s por imagen de validación con C2.

```{admonition} Pendiente
:class: warning
Medición de tiempo y memoria al variar la resolución (128, 256, 512) y el número de regiones (`diseno.md` §3.6);
saldrá de un script de `experiments/` aún no escrito.
```
