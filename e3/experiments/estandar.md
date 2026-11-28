# Estándar del Entregable 3

> **Estado de auditoría local (27/11/2026).** El texto siguiente documenta el plan recuperado, no
> acredita su cumplimiento. El tag declarado no se ha verificado localmente; falta la evaluación GPU
> del original. La discrepancia geométrica quedó resuelta fijando OpenCV 4.11 (21 comparaciones).
> La equivalencia directa con el archivo antiguo de E2 permanece sin verificar. El 4,0 indicado abajo
> es un tope condicionado a R1 y R2 según el enunciado, no una nota garantizada. La potencia y la familia
> completa deben revisarse antes de una campaña confirmatoria. Véase `../../AUDITORIA_CIERRE_E3.txt`.

> **Trazabilidad.** Escrito el 2026-11-23, antes de entrenar el modelo original y antes de las semillas 6 a 10 de los
> rivales. Fuente: enunciado del E3 (p. 3, 21-27, 42 y 47) y `entregable_3/plan.md` (decisiones del 2026-11-23). Se
> congela con la etiqueta de Git `e3-protocolo` junto con `protocolo.md`. No se reduce después.

## 1. El estándar que exige el enunciado (p. 3), sin cambios

El modelo original es superior si cumple a la vez:
1. Una propiedad demostrada formalmente, con supuestos explícitos y verificación numérica.
2. Tres conjuntos de datos adicionales de la misma tarea, de naturaleza distinta, elegidos antes de ver resultados.
3. Superioridad frente a cada uno de los 10 modelos de la familia y a cada baseline fuerte, con Holm sobre toda la
   familia y tres pruebas por contraste, diferencia mayor que δ, |δ de Cliff| ≥ 0,147 e IC inferior mayor que 0.
4. Mejor en la media en al menos 2 de los 3 conjuntos adicionales.
5. Adecuación del modelo (Tablas A y B del E2) y robustez de la conclusión a la prueba elegida.
6. Potencia de 0,90 por contraste y al menos 10 semillas, o las que pida el análisis de potencia si son más.

## 2. Lo que este entregable cumple y lo que no, declarado antes de correr

| Punto | Se hace | No se alcanza | Consecuencia |
|---|---|---|---|
| 1 | Teorema de inversión exacta (`diseno.md` §3.5) con verificación numérica en pytest | — | — |
| 2 y 4 | — | No se corren conjuntos adicionales. El plazo es de una semana y el cómputo estimado es de unas 77 h de L4 (`DECISION_E3_2026-11-21.md`, §4) | Los puntos 2 y 4 quedan incumplidos |
| 3 | Las tres pruebas por contraste con Holm sobre m = 10, en HAZMAT-13 | — | — |
| 6 | 10 semillas para todos los modelos de la familia y para los dos brazos del original | La potencia de 0,90 por contraste no se alcanza con 10 semillas en 7 de los 10 contrastes. Con σ̂_d de planificación, el R necesario va de 10 a 48 (`potencia_e3.json`) | Se reporta la diferencia mínima detectable de cada contraste |

**Por lo tanto, R3 no puede quedar demostrada en este entregable, cualquiera sea el resultado.** El tope de nota es
4,0, declarado de antemano (p. 42). Si el original sale superior en HAZMAT-13 con las tres pruebas, eso se reporta
como evidencia en un solo conjunto, no como R3.

## 3. Hallazgo alternativo (p. 47), declarado antes de congelar

1. **Benchmark riguroso** de las 10 redes con 10 semillas, pruebas pareadas por grupo, Holm, tamaños de efecto y
   potencia reportada.
2. **La limitación de área pequeña** (corregida en la v2): en el análisis de subgrupos del cierre del E2 (fila 67), el
   cuartil de menor área cae en 6 de las 10 redes. En SegFormer-B0, el paper guía, la limitación por área quedó no
   confirmada: su déficit es de nivel, no de área (fila 33). La limitación es común a la familia, no propia del paper
   guía, y se toma como piloto (origen alternativo, p. 8). Se mide si el mecanismo original la reduce frente a su
   propia base (hipótesis del mecanismo).
3. **El teorema de inversión exacta** y la medición del techo geométrico de la representación de núcleo encogido.
