# Pendiente: bases con suscripción (Scopus, Web of Science, IEEE Xplore)

Estas búsquedas las hace una persona en el navegador, con acceso institucional de Uninorte. La revisión en fuentes abiertas
(ver `matriz_novedad.md`) ya se hizo el 2026-11-23; esto la completa y es lo que el enunciado exige como mínimo de tres bases.

## Cómo acceder

- Entrar por la biblioteca de Uninorte, con autenticación propia (la completa la persona, no un programa). Los enlaces por DOI
  se abren anteponiendo `https://doi-org.ezproxy.uninorte.edu.co/` al DOI, por ejemplo
  `https://doi-org.ezproxy.uninorte.edu.co/10.1109/ICASSP49357.2023.10094734`.
- **Scopus:** búsqueda avanzada (Advanced document search). **Web of Science:** Core Collection, búsqueda avanzada
  (Advanced search), Editions: All. **IEEE Xplore:** pestaña Command Search.
- Sin filtro de año, de idioma ni de tipo documental. Si un filtro se cambia, se anota en la hoja.
- Pegar la cadena tal cual. Si la base rechaza la sintaxis, anotar el mensaje exacto y la modificación mínima que se hizo.
- Por cada cadena: anotar la fecha, el número de resultados que muestra la base y revisar el título y el resumen de los
  primeros 50 por relevancia (o todos, si son menos). Marcar como relevante lo que cumple los criterios de `cadenas.md`.
- Exportar (RIS en Scopus, texto plano en Web of Science) solo si el resultado supera los 50 y hace falta revisar el resto.

## Lo más urgente: texto completo de seis trabajos

Cada uno decide un punto abierto de `declaracion.md`. Se buscan por DOI en el proxy.

| Prioridad | Trabajo | DOI | Qué comprobar |
|---|---|---|---|
| 1 | Optimal Kernel for Real-Time Arbitrary-Shaped Text Detection (ICASSP 2023) | 10.1109/ICASSP49357.2023.10094734 | La regla con la que "erosiona los textos según sus características geométricas": si usa el grosor o el radio inscrito de la instancia |
| 2 | Text kernel calculation for arbitrary shape text detection (The Visual Computer 2023) | 10.1007/s00371-023-02963-2 | Cómo se calcula el núcleo y si depende del tamaño o del grosor |
| 3 | Pull Pole Points to Text Contour by Magnetism (IEEE TIP 2025) | 10.1109/TIP.2025.3609196 | Qué propone en lugar del encogimiento uniforme y si alguna parte atenúa la distancia por instancia |
| 4 | Real-Time Text Detection With Similar Mask (IEEE TITS 2024) | 10.1109/TITS.2024.3485061 | Si la máscara "similar" cambia la regla de encogimiento y cómo se reconstruye el contorno |
| 5 | EK-Net++ (Expert Systems with Applications 2024) y Text kernel expansion (Pattern Analysis and Applications 2024) | 10.1016/j.eswa.2024.126159 y 10.1007/s10044-024-01352-2 | Si la distancia de expansión se calcula o se regresa, y si alguna regla de encogimiento es adaptativa |
| 6 | ADNet (IEEE TMM 2023) | 10.1109/TMM.2022.3216729 | Confirmar en el texto lo que se leyó en el código: etiqueta k = D·L'/A' y promedio por región; ver si hay análisis de uniones |

Anotar para cada uno: "antecedente de C1", "antecedente de C2", "parcial" o "sin relación", con la frase o la ecuación que
lo respalda (número de página y de ecuación).

## Cadenas exactas

Las mismas ocho cadenas de `cadenas.md`, escritas para cada base. Se generan con `generar_cadenas_suscripcion.py` a partir
de los mismos grupos de términos; no editar a mano (si hay que cambiar un término, cambiarlo en el script y regenerar).
Las cadenas C1 y C2 buscan el problema, C3 y C8 el mecanismo de encogimiento, C4 el de expansión, C5 el cruce de mecanismo y
problema, C6 el origen y C7 el mecanismo en español.

### C1: problema, EN

Scopus (Advanced search):
```text
TITLE-ABS-KEY((hazmat OR "hazardous material" OR "hazardous materials" OR "dangerous goods") AND (placard OR placards OR label OR labels OR sign OR signs) AND (segmentation OR detection OR "deep learning"))
```

Web of Science Core Collection (Advanced search):
```text
TS=((hazmat OR "hazardous material" OR "hazardous materials" OR "dangerous goods") AND (placard OR placards OR label OR labels OR sign OR signs) AND (segmentation OR detection OR "deep learning"))
```

IEEE Xplore (Command Search, pestaña Command Search):
```text
("All Metadata":hazmat OR "All Metadata":"hazardous material" OR "All Metadata":"hazardous materials" OR "All Metadata":"dangerous goods") AND ("All Metadata":placard OR "All Metadata":placards OR "All Metadata":label OR "All Metadata":labels OR "All Metadata":sign OR "All Metadata":signs) AND ("All Metadata":segmentation OR "All Metadata":detection OR "All Metadata":"deep learning")
```

### C2: problema, ES

Scopus (Advanced search):
```text
TITLE-ABS-KEY(("mercancías peligrosas" OR "materiales peligrosos" OR "sustancias peligrosas") AND (rótulo OR rótulos OR placa OR placas OR etiqueta OR etiquetas) AND (segmentación OR detección OR "aprendizaje profundo"))
```

Web of Science Core Collection (Advanced search):
```text
TS=(("mercancías peligrosas" OR "materiales peligrosos" OR "sustancias peligrosas") AND (rótulo OR rótulos OR placa OR placas OR etiqueta OR etiquetas) AND (segmentación OR detección OR "aprendizaje profundo"))
```

IEEE Xplore (Command Search, pestaña Command Search):
```text
("All Metadata":"mercancías peligrosas" OR "All Metadata":"materiales peligrosos" OR "All Metadata":"sustancias peligrosas") AND ("All Metadata":rótulo OR "All Metadata":rótulos OR "All Metadata":placa OR "All Metadata":placas OR "All Metadata":etiqueta OR "All Metadata":etiquetas) AND ("All Metadata":segmentación OR "All Metadata":detección OR "All Metadata":"aprendizaje profundo")
```

### C3: mecanismo C1: encogimiento adaptativo

Scopus (Advanced search):
```text
TITLE-ABS-KEY(("differentiable binarization" OR DBNet OR "shrink ratio" OR "kernel shrinking" OR "shrunk kernel" OR "shrunk kernels") AND (adaptive OR "instance-adaptive" OR "area-aware" OR "size-adaptive" OR inscribed OR thickness OR "adaptive shrink"))
```

Web of Science Core Collection (Advanced search):
```text
TS=(("differentiable binarization" OR DBNet OR "shrink ratio" OR "kernel shrinking" OR "shrunk kernel" OR "shrunk kernels") AND (adaptive OR "instance-adaptive" OR "area-aware" OR "size-adaptive" OR inscribed OR thickness OR "adaptive shrink"))
```

IEEE Xplore (Command Search, pestaña Command Search):
```text
("All Metadata":"differentiable binarization" OR "All Metadata":DBNet OR "All Metadata":"shrink ratio" OR "All Metadata":"kernel shrinking" OR "All Metadata":"shrunk kernel" OR "All Metadata":"shrunk kernels") AND ("All Metadata":adaptive OR "All Metadata":"instance-adaptive" OR "All Metadata":"area-aware" OR "All Metadata":"size-adaptive" OR "All Metadata":inscribed OR "All Metadata":thickness OR "All Metadata":"adaptive shrink")
```

### C4: mecanismo C2: expansión

Scopus (Advanced search):
```text
TITLE-ABS-KEY((unclip OR "kernel expansion" OR "expansion ratio" OR "learnable expansion" OR "adaptive expansion" OR "adaptive unclip" OR "kernel reconstruction" OR "offset ratio") AND ("text detection" OR "scene text" OR "instance segmentation") AND (shrink OR shrunk OR shrinking OR kernel OR DBNet))
```

Web of Science Core Collection (Advanced search):
```text
TS=((unclip OR "kernel expansion" OR "expansion ratio" OR "learnable expansion" OR "adaptive expansion" OR "adaptive unclip" OR "kernel reconstruction" OR "offset ratio") AND ("text detection" OR "scene text" OR "instance segmentation") AND (shrink OR shrunk OR shrinking OR kernel OR DBNet))
```

IEEE Xplore (Command Search, pestaña Command Search):
```text
("All Metadata":unclip OR "All Metadata":"kernel expansion" OR "All Metadata":"expansion ratio" OR "All Metadata":"learnable expansion" OR "All Metadata":"adaptive expansion" OR "All Metadata":"adaptive unclip" OR "All Metadata":"kernel reconstruction" OR "All Metadata":"offset ratio") AND ("All Metadata":"text detection" OR "All Metadata":"scene text" OR "All Metadata":"instance segmentation") AND ("All Metadata":shrink OR "All Metadata":shrunk OR "All Metadata":shrinking OR "All Metadata":kernel OR "All Metadata":DBNet)
```

### C5: mecanismo + problema

Scopus (Advanced search):
```text
TITLE-ABS-KEY(("differentiable binarization" OR DBNet OR "shrunk kernel" OR "kernel shrinking" OR "progressive scale expansion") AND (hazmat OR placard OR placards OR "dangerous goods" OR hazardous OR "warning sign" OR "warning signs" OR "traffic sign" OR "traffic signs" OR "road sign" OR "road signs" OR pictogram OR pictograms))
```

Web of Science Core Collection (Advanced search):
```text
TS=(("differentiable binarization" OR DBNet OR "shrunk kernel" OR "kernel shrinking" OR "progressive scale expansion") AND (hazmat OR placard OR placards OR "dangerous goods" OR hazardous OR "warning sign" OR "warning signs" OR "traffic sign" OR "traffic signs" OR "road sign" OR "road signs" OR pictogram OR pictograms))
```

IEEE Xplore (Command Search, pestaña Command Search):
```text
("All Metadata":"differentiable binarization" OR "All Metadata":DBNet OR "All Metadata":"shrunk kernel" OR "All Metadata":"kernel shrinking" OR "All Metadata":"progressive scale expansion") AND ("All Metadata":hazmat OR "All Metadata":placard OR "All Metadata":placards OR "All Metadata":"dangerous goods" OR "All Metadata":hazardous OR "All Metadata":"warning sign" OR "All Metadata":"warning signs" OR "All Metadata":"traffic sign" OR "All Metadata":"traffic signs" OR "All Metadata":"road sign" OR "All Metadata":"road signs" OR "All Metadata":pictogram OR "All Metadata":pictograms)
```

### C6: origen: otras aplicaciones

Scopus (Advanced search):
```text
TITLE-ABS-KEY(("differentiable binarization" OR "shrunk kernel" OR "kernel shrinking") AND (segmentation OR "instance segmentation" OR medical OR crack OR defect OR building OR road OR "remote sensing" OR "object detection") AND NOT (" OR t OR e OR x OR t OR   OR d OR e OR t OR e OR c OR t OR i OR o OR n OR "))
```

Web of Science Core Collection (Advanced search):
```text
TS=(("differentiable binarization" OR "shrunk kernel" OR "kernel shrinking") AND (segmentation OR "instance segmentation" OR medical OR crack OR defect OR building OR road OR "remote sensing" OR "object detection") NOT (" OR t OR e OR x OR t OR   OR d OR e OR t OR e OR c OR t OR i OR o OR n OR "))
```

IEEE Xplore (Command Search, pestaña Command Search):
```text
("All Metadata":"differentiable binarization" OR "All Metadata":"shrunk kernel" OR "All Metadata":"kernel shrinking") AND ("All Metadata":segmentation OR "All Metadata":"instance segmentation" OR "All Metadata":medical OR "All Metadata":crack OR "All Metadata":defect OR "All Metadata":building OR "All Metadata":road OR "All Metadata":"remote sensing" OR "All Metadata":"object detection") NOT ("All Metadata":" OR "All Metadata":t OR "All Metadata":e OR "All Metadata":x OR "All Metadata":t OR "All Metadata":  OR "All Metadata":d OR "All Metadata":e OR "All Metadata":t OR "All Metadata":e OR "All Metadata":c OR "All Metadata":t OR "All Metadata":i OR "All Metadata":o OR "All Metadata":n OR "All Metadata":")
```

### C7: mecanismo, ES

Scopus (Advanced search):
```text
TITLE-ABS-KEY(("binarización diferenciable" OR "binarizacion diferenciable" OR "encogimiento del núcleo" OR "detección de texto en escenas" OR "reducción de núcleo"))
```

Web of Science Core Collection (Advanced search):
```text
TS=(("binarización diferenciable" OR "binarizacion diferenciable" OR "encogimiento del núcleo" OR "detección de texto en escenas" OR "reducción de núcleo"))
```

IEEE Xplore (Command Search, pestaña Command Search):
```text
("All Metadata":"binarización diferenciable" OR "All Metadata":"binarizacion diferenciable" OR "All Metadata":"encogimiento del núcleo" OR "All Metadata":"detección de texto en escenas" OR "All Metadata":"reducción de núcleo")
```

### C8: mecanismo C1: distancia, inscrito

Scopus (Advanced search):
```text
TITLE-ABS-KEY(("distance transform" OR "inscribed circle" OR "inscribed radius" OR "medial axis" OR skeleton) AND (shrink OR shrunk OR "kernel map" OR "center region" OR "text region") AND ("instance segmentation" OR "text detection" OR "arbitrary-shaped"))
```

Web of Science Core Collection (Advanced search):
```text
TS=(("distance transform" OR "inscribed circle" OR "inscribed radius" OR "medial axis" OR skeleton) AND (shrink OR shrunk OR "kernel map" OR "center region" OR "text region") AND ("instance segmentation" OR "text detection" OR "arbitrary-shaped"))
```

IEEE Xplore (Command Search, pestaña Command Search):
```text
("All Metadata":"distance transform" OR "All Metadata":"inscribed circle" OR "All Metadata":"inscribed radius" OR "All Metadata":"medial axis" OR "All Metadata":skeleton) AND ("All Metadata":shrink OR "All Metadata":shrunk OR "All Metadata":"kernel map" OR "All Metadata":"center region" OR "All Metadata":"text region") AND ("All Metadata":"instance segmentation" OR "All Metadata":"text detection" OR "All Metadata":"arbitrary-shaped")
```

## Hoja para anotar resultados

Una fila por base y cadena. Fecha de la consulta en formato AAAA-MM-DD. "Relevantes" según los criterios de `cadenas.md`.

| fecha | base | cadena | resultados que muestra la base | revisados | relevantes | trabajos relevantes (título, año, DOI) | amenaza a C1 o C2 (no / parcial / sí) | observación |
|---|---|---|---|---|---|---|---|---|
|  | Scopus | C1 |  |  |  |  |  |  |
|  | Scopus | C2 |  |  |  |  |  |  |
|  | Scopus | C3 |  |  |  |  |  |  |
|  | Scopus | C4 |  |  |  |  |  |  |
|  | Scopus | C5 |  |  |  |  |  |  |
|  | Scopus | C6 |  |  |  |  |  |  |
|  | Scopus | C7 |  |  |  |  |  |  |
|  | Scopus | C8 |  |  |  |  |  |  |
|  | Web of Science | C1 |  |  |  |  |  |  |
|  | Web of Science | C2 |  |  |  |  |  |  |
|  | Web of Science | C3 |  |  |  |  |  |  |
|  | Web of Science | C4 |  |  |  |  |  |  |
|  | Web of Science | C5 |  |  |  |  |  |  |
|  | Web of Science | C6 |  |  |  |  |  |  |
|  | Web of Science | C7 |  |  |  |  |  |  |
|  | Web of Science | C8 |  |  |  |  |  |  |
|  | IEEE Xplore | C1 |  |  |  |  |  |  |
|  | IEEE Xplore | C2 |  |  |  |  |  |  |
|  | IEEE Xplore | C3 |  |  |  |  |  |  |
|  | IEEE Xplore | C4 |  |  |  |  |  |  |
|  | IEEE Xplore | C5 |  |  |  |  |  |  |
|  | IEEE Xplore | C6 |  |  |  |  |  |  |
|  | IEEE Xplore | C7 |  |  |  |  |  |  |
|  | IEEE Xplore | C8 |  |  |  |  |  |  |

## Al terminar

- Pasar la hoja a `matriz_novedad.csv` (mismas columnas) y marcar como cumplida cada base.
- Para cada trabajo nuevo con amenaza "parcial" o "sí": añadirlo a `comparativa.md` y revisar `declaracion.md`.
- Si algún trabajo resuelve d como punto fijo o con una inversa calculada, o atenúa el encogimiento por grosor, avisar antes de
  tocar la declaración: cambia el planteamiento de R1.
