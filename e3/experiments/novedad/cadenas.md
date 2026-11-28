# Cadenas de búsqueda de novedad (R1)

Mecanismo evaluado: encogimiento del núcleo atenuado por el radio inscrito (C1) y expansión consistente
en inferencia como punto fijo con expansión en inglete (C2), sobre la representación de DBNet
(Liao et al., 2020), aplicado a la segmentación binaria de rótulos de mercancías peligrosas.

Cada cadena combina al menos dos de tres ejes: (i) el mecanismo o su familia, (ii) el problema de aplicación,
(iii) sinónimos en inglés y en español. Las cadenas C3, C4, C6 y C8 buscan el mecanismo en otras aplicaciones
para identificar su origen y a quién se le ocurrió antes. La sintaxis de las fuentes abiertas es la que ejecuta
`buscar.py` (el código es la fuente de verdad de lo que se envió); la de Scopus, Web of Science e IEEE Xplore
está en `pendiente_bases_suscripcion.md`.

| Id | Eje | Qué busca | Idioma |
|---|---|---|---|
| C1 | problema | rótulos, placas y etiquetas de mercancías peligrosas con segmentación o detección | EN |
| C2 | problema | lo mismo en español (mercancías, materiales y sustancias peligrosas; rótulo, placa, etiqueta) | ES |
| C3 | mecanismo (C1) | DBNet o núcleo encogido con razón de encogimiento adaptativa por tamaño, grosor o radio inscrito | EN |
| C4 | mecanismo (C2) | expansión, unclip u offset del núcleo adaptativo, aprendido, inverso o reconstruido | EN |
| C5 | mecanismo + problema | DBNet, PSENet o encogimiento de núcleo aplicado a placas hazmat, señales de tráfico o de advertencia | EN |
| C6 | origen | DBNet y núcleos encogidos en aplicaciones que no son texto (médicas, grietas, defectos, teledetección) | EN |
| C7 | mecanismo | binarización diferenciable, encogimiento del núcleo, detección de texto en escena | ES |
| C8 | mecanismo (C1) | etiqueta por transformada de distancia, círculo inscrito o eje medial con encogimiento de núcleo | EN |

## Texto de cada cadena por fuente abierta

Las cadenas completas, tal como se enviaron a OpenAlex, arXiv, Semantic Scholar (búsqueda masiva), OpenReview,
GitHub y Hugging Face Hub, están en el diccionario `CADENAS` de `buscar.py`. Resumen de la lógica:

- **OpenAlex** (`/works?search=`, booleano con AND, OR, NOT y comillas): título, resumen y texto completo indexado.
- **arXiv** (`export.arxiv.org/api/query`, `all:`): todos los campos de los metadatos.
- **Semantic Scholar** (`/graph/v1/paper/search/bulk`, `+` y `|`): título y resumen.
- **OpenReview** (`api2.openreview.net/notes/search?term=`): búsqueda de texto libre, sin booleanos; sirve como complemento.
- **GitHub** (`/search/repositories`) y **Hugging Face Hub** (`/api/models`, `/datasets`, `/spaces`): palabras clave
  del mecanismo y del problema; revisión de código, no de literatura.
- **Google Scholar**: no accesible desde el cliente automático (HTTP 403 el 2026-11-23). Queda para el
  verificador o para la persona con navegador.
- **Papers with Code**: el 2026-11-23 redirige a `huggingface.co/papers/trending` (el sitio quedó integrado
  en Hugging Face). Las tablas de resultados se revisaron en los repositorios y en Hugging Face Papers.

## Criterios de cribado (fijados antes de mirar los resultados)

- **Revisado:** título y, si tiene, resumen leídos. Se revisan los primeros 50 por relevancia (arXiv, OpenAlex,
  OpenReview, GitHub) o por número de citas (Semantic Scholar masiva, que no ofrece orden por relevancia).
  Cuando la fuente declara más de 50, el resto no se revisó y se anota.
- **Relevante:** el trabajo (a) segmenta o detecta rótulos o placas de mercancía peligrosa, o (b) usa una
  representación de núcleo encogido y expansión (DBNet, PSENet, PAN y derivados) con una regla de encogimiento
  o de expansión que dependa de la instancia, o (c) fija con una inversa la expansión del núcleo.
  Las menciones de "hazmat" en logística, riesgo o derecho no cuentan.
- **Amenaza a C1 o C2:** un trabajo que (C1) calcule la razón o la distancia de encogimiento con el grosor o el
  radio inscrito de la instancia, o la atenúe en las instancias pequeñas, o (C2) obtenga la distancia de
  expansión como punto fijo de la regla de encogimiento, o como inversa calculada de ella sin cabeza de
  regresión.
