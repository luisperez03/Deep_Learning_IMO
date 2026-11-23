# Criterios de elegibilidad aplicados en el cribado

Fijados el 22/11/2026, antes de revisar `cribado_manual.csv`. Concretan los pasos 3-5 del cribado de `PROTOCOLO_BUSQUEDA.md`
para las dos partes del cierre: la muestra de validación del precribado (título y resumen) y el núcleo de dominio
(texto completo).

## Muestra del precribado: título y resumen

Pregunta: ¿este registro debía pasar a texto completo según el protocolo? `incluir` si cumple las cuatro condiciones:

1. **Método aprendido de segmentación.** Propone o evalúa una red que predice una máscara por píxel (semántica, de
   instancia o panóptica).
2. **Entrada transferible.** Imagen 2D, y el aporte no depende de algo exclusivo de su modalidad: volumen 3D, nube de
   puntos, señal 1D o bandas espectrales. Las imágenes médicas 2D cuentan si el mecanismo no es propio de la
   modalidad.
3. **Aporte pertinente.** Una arquitectura candidata o de su misma línea, un mecanismo dirigido a una dificultad de
   HAZMAT-13 (objetos pequeños, bordes, multiescala, contexto global, eficiencia), o una comparación de arquitecturas
   de segmentación.
4. **Método original.** No es revisión de congreso, errata, editorial ni revisión de literatura.

Si falla alguna, `excluir` con el motivo del protocolo:

| Motivo | Cuándo |
|---|---|
| tarea sin transferencia | Falla 1, 2 o 3 |
| sin método original | Falla 4: revisiones de literatura, resúmenes de congreso o editoriales |
| información insuficiente | No hay resumen y el título no basta para decidir |
| duplicado o versión | El mismo estudio que otro grupo |
| fuera de ventana | Anterior a 2019, sin función de fundamento |
| idioma no evaluable | Ni inglés ni español |

## Núcleo de dominio: texto completo

`incluir` si el estudio trata rótulos, placas o señales de mercancía peligrosa (o etiquetas de peligro equivalentes)
en imagen, con cualquier tarea: detección, reconocimiento, localización o segmentación. Esto da el denominador del gap.
`excluir` con motivo si el objeto no es ese, por ejemplo "materiales peligrosos" en otro sentido o un vehículo sin
rótulo. Para cada incluido se registran dos preguntas que pueden refutar el gap:

- `segmenta_rotulo_por_pixel`: ¿entrena una red profunda que produce una máscara por píxel del rótulo?
- `compara_arquitecturas_con_semillas`: ¿compara dos o más arquitecturas de segmentación sobre ese objeto e informa la
  variabilidad entre corridas?

Si no se recupera el texto completo, se decide con el resumen y se anota en `nota`. El motivo "texto completo no
recuperado" se reserva para cuando el resumen tampoco basta.

## Bloque transversal (Q5 y Q6): censo a título y resumen

Añadido el 22/11/2026 al reabrir Q5 y Q6, antes de revisar sus registros. Estas consultas buscan mecanismos nacidos en
secuencias o señales (modelos de espacio de estados y Mamba; convoluciones dilatadas o temporales) y su paso a visión.
`incluir` (clasificación `mecanismo transferible`) si el estudio aplica o transfiere ese mecanismo a una tarea de visión
sobre imágenes 2D o fotogramas: segmentación, detección o clasificación. Es la evidencia publicada de transferencia que
pide la ficha de mecanismo. `excluir` con motivo si:

- el mecanismo se queda en su dominio de origen (pronóstico de series, voz, audio, texto o señales 1D): `tarea sin
  transferencia`;
- la imagen es solo un medio y el aporte es propio de otra modalidad (espectrogramas tratados como señal, radar o SAR
  multitemporal donde lo que se modela es la serie): `tarea sin transferencia`;
- es un resumen de congreso, una errata o una revisión: `sin método original`.

## Quién decide

Las decisiones las propone un asistente de IA y las valida el investigador (protocolo, paso 6). No hay segundo
revisor humano, así que no se calcula kappa.
