## De un píxel RGB a un mapa de características

La entrada tiene forma $3\times256\times256$. En cada posición $(u,v)$ el filtro $j$ lee la misma vecindad de 3 × 3 en todos los canales. En la implementación de PyTorch, la operación llamada convolución es una correlación cruzada con pesos aprendidos:

$$
a_{j,u,v}=b_j+\sum_{c}\sum_{r,s=-1}^{1}W_{j,c,r,s}\,x_{c,u+r,v+s},
\qquad h_{j,u,v}=\max(0,a_{j,u,v}).
$$

El mismo peso interviene en muchas posiciones. Así, un patrón local puede activar el filtro en diferentes partes de la imagen. El relleno de un píxel de la convolución conserva ancho y alto. Es distinto del relleno del preprocesamiento que ajusta la fotografía al lienzo cuadrado. En `block`, cada una de las dos convoluciones va seguida de ReLU.

![Un filtro de tres por tres detecta una transición y ReLU elimina la respuesta negativa](figuras/filtro_convolucion.png)

En la primera ventana, cada fila aporta −1 × 0 + 0 × 1 + 1 × 1 = 1. Al sumar tres filas se obtiene 3. Al deslizar el mismo filtro un píxel a la derecha se obtiene −3 y ReLU devuelve 0. Es un ejemplo didáctico de un canal con un filtro construido a mano. En las tres redes, cada filtro combina todos los canales de entrada y aprende sus coeficientes durante el entrenamiento.

### Qué información llega al decodificador

El codificador transforma $16\times256\times256$ en $32\times128\times128$ y luego en $64\times64\times64$. En cada reducción, `max_pool2d` conserva el máximo de cada ventana de 2 × 2. Las activaciones profundas combinan información de una región mayor de la fotografía, pero ya no conservan una muestra por cada posición original.

| Red y método en el código | Camino de vuelta a 256 × 256 | Consecuencia |
|---|---|---|
| `SmallFCN.forward` | Una convolución 1 × 1 lleva los 64 canales a un logit a resolución 64 × 64. La interpolación bilineal amplía ese mapa directamente. | No recibe mapas laterales del codificador. Un logit de salida combina vecinos del mapa grueso. |
| `SmallSegNet.forward` | Reduce 64 a 32 canales con una convolución 1 × 1. Dos operaciones de *unpooling* devuelven los valores a los índices de máximos guardados, seguidas por bloques convolucionales. | El índice recupera una posición, no los valores descartados. Las convoluciones del decodificador completan el mapa disperso. |
| `SmallUNet.forward` | Amplía por vecino más cercano y concatena el mapa profundo con 32 canales del codificador (96 en total). Repite la operación con 16 canales (48 en total). | El decodificador recibe las activaciones superficiales completas en sus posiciones originales de cada nivel. |

Las dos últimas redes producen el logit final mediante una cabeza 1 × 1 a resolución completa. Ninguna emplea transposición convolucional, pesos preentrenados ni normalización por lotes. La comparación empírica decide si estas diferencias ayudan en este corpus.

## Del error observado al gradiente

Sea $V$ el conjunto de píxeles válidos de una imagen, $n=|V|$, $z_i$ su logit, $p_i=(1+e^{-z_i})^{-1}$ y $y_i\in\{0,1\}$. En `colab_runner.loss_fn` se indexa por $V$ antes de reducir. La BCE se calcula directamente con logits para mantener estabilidad numérica. Para explicar el cálculo puede escribirse en probabilidades:

$$
L_{\rm BCE}=-\frac1n\sum_{i\in V}\big[y_i\log p_i+(1-y_i)\log(1-p_i)\big].
$$

Sea $A=2\sum_{i\in V}p_i y_i+\varepsilon$ y $B=\sum_{i\in V}(p_i+y_i)+\varepsilon$, con $\varepsilon=10^{-6}$. La pérdida usada fue $L=(L_{\rm BCE}+1-A/B)/2$. La BCE penaliza errores por píxel. El término de solapamiento vincula los píxeles de la imagen mediante un denominador común y da peso a la pequeña región de etiqueta.

Antes de aplicar umbral, la derivada respecto de un logit válido es

$$
\frac{\partial L}{\partial z_i}
=\tfrac12\left[\frac{p_i-y_i}{n}
+\frac{A-2y_i B}{B^2}p_i(1-p_i)\right].
$$

Por la regla de la cadena, si $\delta_{j,u,v}$ es el gradiente que llega a la preactivación $a_{j,u,v}$, entonces $\partial L/\partial W_{j,c,r,s}=\sum_{u,v}\delta_{j,u,v} x_{c,u+r,v+s}$. La suma recoge todos los usos del filtro. ReLU transmite el gradiente para preactivaciones positivas y *max pooling* lo dirige a la posición elegida. Las concatenaciones de U-Net dan al gradiente más de una ruta hacia las primeras capas. Los píxeles del relleno no aportan términos a la pérdida.

### Un paso de Adam

Por cada imagen, `run` limpia los gradientes con `zero_grad`, ejecuta el modelo, evalúa la pérdida y llama a `loss.backward()`. PyTorch obtiene $g_t=\nabla_\theta L_t$ para cada parámetro por diferenciación automática. `optimizer.step()` aplica Adam con tasa $\eta=0{,}001$, momentos iniciales cero, $\beta_1=0{,}9$, $\beta_2=0{,}999$, $\epsilon_A=10^{-8}$ y sin decaimiento de pesos:

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,
$$

$$
\widehat m_t=\frac{m_t}{1-\beta_1^t},\quad
\widehat v_t=\frac{v_t}{1-\beta_2^t},\quad
\theta_t=\theta_{t-1}-\eta\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon_A}.
$$

Las operaciones son por coordenada. El primer momento suaviza la dirección del gradiente y el segundo adapta la escala del paso. El $\epsilon_A$ de Adam cumple una función distinta del suavizado del Dice. La validación selecciona pesos al final de cada época; no produce actualizaciones del optimizador. [Kingma y Ba, *Adam: A Method for Stochastic Optimization*](https://arxiv.org/abs/1412.6980)

## Del logit a Dice e IoU

En `dataset_ready.metrics`, $p_i\geq0{,}5$ define la máscara predicha $P$. Sea $Y$ la referencia binaria en el área válida. Los aciertos positivos son $|P\cap Y|$; la unión $|P\cup Y|$ incluye además falsos positivos y negativos. Por imagen se cumple

$$
J=\mathrm{IoU}=\frac{VP}{VP+FP+FN},\qquad
D=\mathrm{Dice}=\frac{2VP}{2VP+FP+FN}=\frac{2J}{1+J}.
$$

Por ejemplo, con 8 píxeles positivos correctos, 1 falso positivo y 3 falsos negativos, IoU es $8/12$ y Dice es $16/20$. Para coincidencia parcial Dice resulta mayor; ambos valen uno al coincidir perfectamente. El código asigna uno también a dos máscaras vacías. La relación es no lineal: el Dice promedio de la tabla no se obtiene transformando la IoU promedio. El Dice suave de entrenamiento usa probabilidades continuas; estas métricas de evaluación usan máscaras umbralizadas.
