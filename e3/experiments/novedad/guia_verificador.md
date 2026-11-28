# Guía del verificador de la revisión de novedad

Propósito: que una persona ajena al diseño repita al menos dos cadenas y emita su concepto sobre la declaración de novedad.
Tiempo estimado: 60 a 90 minutos. Requiere acceso institucional a Scopus o Web of Science (y, si es posible, IEEE Xplore).

## Qué se verifica

El diseño propone dos cambios sobre la detección de texto en escena con binarización diferenciable (DBNet) y los aplica a
la segmentación de rótulos de mercancías peligrosas: **(C1)** encoger menos el núcleo de las regiones pequeñas, según su
radio inscrito, y **(C2)** recuperar el contorno en inferencia con una distancia calculada, no fija, que deshace exactamente el
encogimiento. La declaración dice que ningún estudio aplica este modelo a rótulos de mercancías peligrosas.

## Qué hacer

1. **Cadenas obligatorias: C3 y C5.** Están en `pendiente_bases_suscripcion.md`, listas para pegar. C3 busca el mecanismo de
   encogimiento adaptativo; C5 busca el mecanismo aplicado a rótulos o señales. Opcional: C4 (expansión).
2. **Dónde.** Scopus y Web of Science (Core Collection, Editions: All); en IEEE Xplore, la cadena C5 en Command Search.
   Sin filtros de año ni de tipo.
3. **Cómo.** Pegar la cadena tal cual. Si la base la rechaza, anotar el mensaje y la modificación mínima. No añadir ni quitar
   términos sin anotarlo.
4. **Revisión propia.** Leer el título y el resumen de los primeros 50 resultados. Marcar como relevante un trabajo que:
   - segmente o detecte rótulos o placas de mercancía peligrosa; o
   - use un núcleo encogido con expansión (DBNet, PSENet y derivados) con una regla de encogimiento o de expansión que dependa
     de la instancia (tamaño, grosor, radio inscrito); o
   - calcule la distancia de expansión como inversa o punto fijo de la regla de encogimiento, sin regresarla.
   El criterio es el de la persona que verifica; no se compara con la matriz hasta terminar.
5. **Comparar al final.** Abrir `matriz_novedad.md` y `comparativa.md` y ver si los relevantes propios ya están. Todo trabajo
   relevante que no esté se anota aparte, con título, año y DOI.

## Qué anotar

Por cada cadena y base: fecha, texto exacto enviado, número de resultados que muestra la base, cuántos se revisaron, cuántos
se marcaron relevantes y cuáles (título, año, DOI). Si un resultado le parece un antecedente directo de C1 o de C2, anotar la
frase del resumen que lo sustenta.

## Cómo emitir el concepto

Un párrafo corto, firmado y fechado, con una de tres conclusiones y su razón:

- **Coincide.** Con las cadenas repetidas no se encontró ningún estudio que aplique el modelo propuesto a rótulos de
  mercancías peligrosas ni que haga C1 o C2 tal como se describen. Se indica qué trabajos cercanos aparecieron.
- **Hay un hallazgo.** Se encontró al menos un estudio que anticipa C1, C2 o la aplicación. Se lista con DOI y se dice qué
  parte anticipa (nada, una parte o todo).
- **No concluyente.** No se pudo ejecutar alguna cadena, o un resultado depende de un texto completo que no se leyó. Se dice
  cuál.

Formato sugerido:

> Concepto del verificador. Fecha: ____. Bases y cadenas repetidas: ____. Resultados observados (por cadena y base): ____.
> Relevantes propios: ____. Relevantes que no estaban en la matriz: ____. Conclusión (coincide, hay un hallazgo o no
> concluyente): ____. Razón en una o dos frases: ____. Nombre y afiliación: ____.

El concepto se anexa a la entrega junto con la hoja de resultados.
