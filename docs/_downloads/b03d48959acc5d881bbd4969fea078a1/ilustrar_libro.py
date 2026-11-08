"""Ilustraciones del informe. Usa datos de train y tensores sin datos para las redes."""
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

from inicio_voc import PROJECT, palette, sha
from modelos_base import MODELS
from preparar_base import read_csv

BOOK = PROJECT / "libro"
OUT = BOOK / "figuras"
INK, TEAL, GOLD = "#16374a", "#09675e", "#965321"
PAPER, MINT, SAND = "#f7faf9", "#dcefeb", "#fff0da"
NAMES = {"fcn":"FCN", "segnet":"SegNet", "unet":"U-Net"}
SPANISH = {
    "poison":"Tóxico", "oxygen":"Oxígeno", "flammable":"Inflamable",
    "flammable-solid":"Sólido inflamable", "corrosive":"Corrosivo",
    "dangerous":"Peligroso", "non-flammable-gas":"Gas no inflamable",
    "organic-peroxide":"Peróxido orgánico", "explosive":"Explosivo",
    "radioactive":"Radiactivo", "inhalation-hazard":"Peligro por inhalación",
    "spontaneously-combustible":"Combustión espontánea",
    "infectious-substance":"Sustancia infecciosa",
}


def font(size, bold=False):
    names = ["segoeuib.ttf","DejaVuSans-Bold.ttf"] if bold else ["segoeui.ttf","DejaVuSans.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name,size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def card(draw, box, title, detail, color=MINT):
    x,y,right,bottom = box
    draw.rounded_rectangle(box, radius=14, fill=color, outline="#adc5bf", width=2)
    center = (x+right)//2
    draw.text((center,y+14),title,font=font(23,True),fill=INK,anchor="mt")
    draw.text((center,bottom-15),detail,font=font(21),fill=INK,anchor="mb")


def arrow(draw, path, color=TEAL):
    draw.line(path, fill=color, width=4)
    x,y = path[-1]
    px,py = path[-2]
    dx,dy = np.sign(x-px),np.sign(y-py)
    draw.polygon([(x,y),(x-dx*11-dy*6,y-dy*11+dx*6),(x-dx*11+dy*6,y-dy*11-dx*6)],fill=color)


def workflow():
    im = Image.new("RGB",(1110,640),PAPER)
    d = ImageDraw.Draw(im)
    d.text((35,22),"De la fotografía a la evaluación",font=font(37,True),fill=INK)
    card(d,(35,95,455,178),"1.685 fotografías","1.583 con máscara utilizable")
    card(d,(615,95,1075,178),"1.451 pares elegibles","132 pares en cuarentena")
    arrow(d,[(456,136),(614,136)])
    d.text((535,111),"revisión",font=font(18),anchor="mm",fill=TEAL)
    d.text((554,211),"Asignación de grupos completos",font=font(23,True),anchor="mm",fill=TEAL)
    d.line([(846,179),(846,235),(194,235)],fill=TEAL,width=4)
    d.line([(846,235),(916,235)],fill=TEAL,width=4)
    for cx in [194,555,916]:
        arrow(d,[(cx,235),(cx,265)])
    for cx,title,sub in [(194,"Entrenamiento","1.015 imágenes · 567 grupos"),(555,"Validación","218 imágenes · 122 grupos"),(916,"Test reservado","218 imágenes · 122 grupos")]:
        card(d,(cx-158,266,cx+158,349),title,sub)
    arrow(d,[(194,350),(194,402)])
    arrow(d,[(555,350),(555,402)])
    card(d,(36,403,352,490),"Ajustar pesos","3 redes × 5 semillas",SAND)
    card(d,(397,403,713,490),"Elegir pesos","Mayor Dice por grupo",SAND)
    arrow(d,[(353,447),(396,447)],GOLD)
    arrow(d,[(916,350),(916,402)])
    card(d,(758,403,1074,490),"Evaluar una vez","Por corrida · umbral 0,5",SAND)
    arrow(d,[(714,447),(757,447)],GOLD)
    arrow(d,[(916,491),(916,537)],GOLD)
    card(d,(340,538,1074,621),"3.270 evaluaciones guardadas","15 corridas × 218 imágenes; comparación pareada por grupo")
    im.save(OUT/"flujo_experimento.png")


def network_map(name):
    im = Image.new("RGB",(1040,566),PAPER)
    d = ImageDraw.Draw(im)
    d.text((28,18),NAMES[name],font=font(36,True),fill=INK)
    d.text((27,70),"Entrada RGB · 3 × 256 × 256",font=font(21),fill=TEAL)
    d.text((682,70),"Salida · 1 × 256 × 256",font=font(21),fill=GOLD)
    left,right = (26,347),(681,1014)
    rows = [115,268,421]
    for i,(channels,side) in enumerate([(16,256),(32,128),(64,64)]):
        y=rows[i]
        card(d,(left[0],y,left[1],y+81),f"Bloque {i+1} · 2 conv.",f"{channels} × {side} × {side}")
        if i<2:
            arrow(d,[(185,y+82),(185,y+151)])
            d.text((202,y+121),"pool 2 × 2",font=font(18),fill=TEAL)
    if name=="fcn":
        card(d,(right[0],421,right[1],502),"Cabeza 1 × 1","1 × 64 × 64",SAND)
        arrow(d,[(348,462),(680,462)])
        card(d,(right[0],115,right[1],196),"Ampliación bilineal","1 × 256 × 256",SAND)
        arrow(d,[(848,420),(848,197)],GOLD)
        d.text((507,275),"Sin conexiones\nlaterales",anchor="mm",align="center",font=font(24),fill=TEAL)
    else:
        seg = name=="segnet"
        labels = [("Bloque + cabeza 1 × 1","16 → 1 · 256 × 256"),
                  ("Bloque decodificador",("16" if seg else "32")+" × 128 × 128"),
                  ("Reducción 1 × 1","64 → 32 · 64 × 64") if seg else ("Mapa profundo","64 × 64 × 64")]
        for y,(title,detail) in zip(rows,labels):
            card(d,(right[0],y,right[1],y+81),title,detail,SAND)
        arrow(d,[(348,462),(680,462)])
        for y in [115,268]:
            arrow(d,[(848,y+152),(848,y+82)],GOLD)
            d.text((862,y+100),"unpool" if seg else "ampliar",font=font(18),fill=GOLD)
            # La rama sale del pooling (índices) o del bloque (mapa completo).
            start_y = y+111 if seg else y+40
            start_x = 185 if seg else 348
            path=([(start_x,start_y),(847,start_y)] if seg else
                  [(start_x,start_y),(680,start_y)])
            arrow(d,path,TEAL)
            d.text((566,start_y-18),"Índices" if seg else "Concatenar",font=font(19,True),anchor="mm",fill=TEAL)
    d.text((28,530),"Dimensiones: canales × alto × ancho",font=font(18),fill=INK)
    im.save(OUT/f"mapa_{name}.png")


def inventory_layers():
    sections = []
    summary = []
    for name,constructor in MODELS.items():
        layers = []
        def record(module, inputs, output):
            layers.append((module,output.shape[1:],output.numel()))
        # El dispositivo meta resuelve dimensiones sin asignar pesos ni evaluar fotografías.
        with torch.device("meta"):
            model = constructor()
            convs = {module:key for key,module in model.named_modules() if isinstance(module,torch.nn.Conv2d)}
            hooks = [module.register_forward_hook(record) for module in convs]
            model(torch.empty(1,3,256,256))
            for hook in hooks:
                hook.remove()
        n = lambda x: f"{x:,}".replace(",",".")
        total = sum(v.numel() for v in model.parameters())
        units = sum(row[2] for row in layers)
        filters = sum(m.out_channels for m,_,_ in layers)
        summary.append(f"| {NAMES[name]} | {len(layers)} | {filters} | {n(units)} | {n(total)} |")
        sections += [f"### {NAMES[name]}", f"![Camino del codificador y reconstrucción espacial de {NAMES[name]}](figuras/mapa_{name}.png)", {
            "fcn":"La cabeza calcula un solo logit por posición en el mapa de 64 × 64. La ampliación bilineal recupera el tamaño de la máscara sin añadir pesos.",
            "segnet":"Las ramas laterales transportan los índices del máximo de cada ventana. El decodificador usa estos índices para recolocar los valores antes de aplicar sus convoluciones.",
            "unet":"Las ramas laterales copian activaciones completas. Tras ampliar por vecino más cercano, se concatenan 64 + 32 canales a 128 × 128 y 32 + 16 canales a 256 × 256."
        }[name]]
        table=["| Capa en PyTorch | Núcleo | Canales entrada → salida | Salida (C × H × W) | Parámetros |",
               "|---|---|---:|---|---:|"]
        for m,shape,_ in layers:
            table.append(f"| `{convs[m]}` | {m.kernel_size[0]} × {m.kernel_size[1]} | {m.in_channels} → {m.out_channels} | {' × '.join(map(str,shape))} | {n(sum(p.numel() for p in m.parameters()))} |")
        sections.append("\n".join([f"```{{dropdown}} Ver las {len(layers)} capas convolucionales de {NAMES[name]}","",*table,"```"]))
    intro = ["### Cómo contar capas y neuronas",
             "Contamos como capa entrenable cada convolución, incluida la cabeza final de 1 × 1. Un filtro produce un canal y reutiliza los mismos pesos en muchas posiciones. Por eso conviene distinguir filtros, activaciones y parámetros. Por ejemplo, la primera convolución de las tres redes tiene 16 filtros, produce 16 × 256 × 256 = 1.048.576 activaciones por imagen y aprende (3 × 3 × 3 + 1) × 16 = 448 parámetros.",
             "Si por «neuronas» se entienden las **unidades de salida de las convoluciones**, la columna de activaciones suma esas unidades en todas las capas por imagen. Depende de la resolución. Los parámetros son los pesos y sesgos aprendidos, compartidos entre posiciones. Pooling, ReLU, ampliaciones y concatenaciones intervienen en el cálculo y no añaden parámetros.",
             "\n".join(["| Red | Capas entrenables | Filtros sumados | Activaciones convolucionales por imagen | Parámetros |","|---|---:|---:|---:|---:|",*summary])]
    (BOOK/"mapas_redes.inc").write_text("\n\n".join(intro+sections)+"\n",encoding="utf-8")


def convolution_example():
    x=np.tile([0,1,1,0],(4,1))
    w=np.tile([-1,0,1],(3,1))
    windows=np.lib.stride_tricks.sliding_window_view(x,w.shape)
    z=(windows*w).sum(axis=(-2,-1))
    canvas=Image.new("RGB",(1100,292),PAPER)
    d=ImageDraw.Draw(canvas)
    for start,title,values in [(28,"Entrada",x),(282,"Filtro",w),(545,"Suma ponderada",z),(834,"Tras ReLU",np.maximum(z,0))]:
        d.text((start,20),title,font=font(23,True),fill=INK)
        for (r,c),val in np.ndenumerate(values):
            left,top=start+46*c,70+46*r
            color=SAND if val<0 else MINT if val>0 else "#edf1f2"
            d.rectangle((left,top,left+44,top+44),fill=color)
            d.text((left+22,top+22),str(val),anchor="mm",font=font(21),fill=INK)
    d.text((237,110),"×",font=font(30),fill=TEAL)
    arrow(d,[(454,122),(524,122)])
    arrow(d,[(683,122),(813,122)])
    d.text((28,260),"Un canal · sesgo cero · paso de un píxel · sin relleno en este ejemplo",font=font(18),fill=TEAL)
    canvas.save(OUT/"filtro_convolucion.png")


def gallery():
    folder=OUT/"clases"
    folder.mkdir(exist_ok=True)
    inv={r["id"]:r for r in read_csv(PROJECT/"resultados/inicio_voc/inventario.csv")}
    train=read_csv(PROJECT/"resultados/preentrenamiento/train.csv")
    colors={name:rgb for rgb,name in palette(PROJECT.parent/"VOC2012/labelmap.txt").items() if name!="background"}
    assert set(colors)==set(SPANISH)
    content=["::::{grid} 1 2 2 3",":gutter: 2",""]
    for name,translation in SPANISH.items():
        candidates=[r for r in train if inv[r["id"]]["classes"]==name]
        row=max(candidates,key=lambda r:(float(inv[r["id"]]["foreground_fraction"]),r["id"]))
        path=PROJECT.parent/row["original_mask_path"]
        photo=PROJECT.parent/row["image_path"]
        assert sha(path)==row["original_mask_sha256"] and sha(photo)==row["image_sha256"]
        with Image.open(path) as original:
            mask=np.all(np.array(original.convert("RGB"))==colors[name],axis=-1)
        y,x=np.nonzero(mask)
        width,height=int(x.max()-x.min()+1),int(y.max()-y.min()+1)
        pad=max(8,round(.16*max(width,height)))
        with Image.open(photo) as source:
            box=(max(0,int(x.min())-pad),max(0,int(y.min())-pad),min(source.width,int(x.max())+1+pad),min(source.height,int(y.max())+1+pad))
            crop=source.convert("RGB").crop(box)
            card_image=ImageOps.pad(crop,(420,300),method=Image.Resampling.LANCZOS,color="#e9eeeb")
            card_image.save(folder/(name+".jpg"),quality=90)
        content.extend([f":::{{grid-item-card}} {translation}",f"![Recorte anotado como {name} en {row['id']}](figuras/clases/{name}.jpg)","",f"`{name}` · {row['id']}",":::",""])
    content.append("::::")
    (BOOK/"galeria_clases.inc").write_text("\n".join(content)+"\n",encoding="utf-8")


def class_counts():
    initial = PROJECT/"resultados/inicio_voc"
    prep = PROJECT/"resultados/preentrenamiento"
    manifest = {r["id"]:r for r in read_csv(prep/"manifest.csv")}
    audited = {r["class"]:int(r["images"]) for r in read_csv(initial/"distribucion_clases.csv")}
    parts = ["train","val","test","quarantine"]
    counts = {name:dict.fromkeys(parts,0) for name in SPANISH}
    for row in read_csv(initial/"inventario.csv"):
        if row["status"] != "ok":
            continue
        record = manifest[row["id"]]
        part = record["split"] or record["status"]
        for name in set(row["classes"].split("|")):
            counts[name][part] += 1
    lines = ["| Categoría original | Total con máscara | Entrenamiento | Validación | Test | Cuarentena |",
             "|---|---:|---:|---:|---:|---:|"]
    for name in sorted(counts,key=lambda k:-audited[k]):
        values = [counts[name][p] for p in parts]
        assert sum(values)==audited[name],name
        lines.append("| " + " | ".join(map(str,[SPANISH[name],audited[name],*values])) + " |")
    (BOOK/"conteo_clases.inc").write_text("\n".join(lines)+"\n",encoding="utf-8")


def binary_example():
    row = next(r for r in read_csv(PROJECT/"resultados/preentrenamiento/train.csv") if r["id"]=="image_1165")
    paths = [PROJECT.parent/row["image_path"],PROJECT.parent/row["original_mask_path"],PROJECT/row["binary_mask_path"]]
    images = []
    for path,key in zip(paths,["image_sha256","original_mask_sha256","binary_mask_sha256"]):
        assert sha(path)==row[key]
        with Image.open(path) as source:
            images.append(source.convert("RGB") if key!="binary_mask_sha256" else source.convert("L"))
    rgb = np.asarray(images[1])
    y = np.asarray(images[2])
    assert np.array_equal(y,rgb.any(axis=2))
    palette_rgb = palette(PROJECT.parent/"VOC2012/labelmap.txt")
    display = np.zeros_like(rgb)
    for color,name in palette_rgb.items():
        if name in ("flammable-solid","radioactive"):
            tint = (234,157,71) if name=="flammable-solid" else (97,184,204)
            display[np.all(rgb==color,axis=2)] = tint
    assert np.array_equal(display.any(axis=2),y)
    canvas = Image.new("RGB",(1040,430),PAPER)
    d=ImageDraw.Draw(canvas)
    for col,(title,im) in enumerate(zip(["Fotografía","Categorías originales","Objetivo binario"],
                                        [images[0],Image.fromarray(display),Image.fromarray(y*255)])):
        x=18+col*340
        d.text((x,16),title,font=font(25,True),fill=INK)
        canvas.paste(ImageOps.pad(im.convert("RGB"),(320,300),method=Image.Resampling.NEAREST if col else Image.Resampling.LANCZOS,color="#e7edeb"),(x,60))
    d.text((18,382),"Train · image_1165",font=font(18),fill=INK)
    for x,color,label in [(357,"#ea9d47","Sólido inflamable"),(517,"#61b8cc","Radiactivo")]:
        d.rectangle((x,385,x+13,398),fill=color)
        d.text((x+18,380),label,font=font(16),fill=INK)
    d.text((698,382),"Blanco: etiqueta   ·   Negro: fondo",font=font(18),fill=INK)
    canvas.save(OUT/"objetivo_binario.png")


if __name__=="__main__":
    OUT.mkdir(exist_ok=True)
    workflow()
    for key in MODELS:
        network_map(key)
    inventory_layers()
    convolution_example()
    gallery()
    class_counts()
    binary_example()
