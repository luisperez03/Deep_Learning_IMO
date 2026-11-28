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
