"""Gráficos estáticos exportables sin dependencias adicionales a Pillow/NumPy.

Separar el dibujo del análisis facilita leer el cuaderno: las celdas muestran
qué se calcula y aquí se controla únicamente cómo se representa.
"""
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import unicodedata

BLUE='#24778a'; ORANGE='#d47437'; DARK='#163345'

class PortableDraw:
    """Evita glifos ausentes en la fuente incorporada; texto gráfico sin tildes.

    Los comentarios y el cuaderno conservan español con tildes. No depender de
    una fuente instalada mantiene idénticas las imágenes al regenerarlas.
    """
    def __init__(self,image):self.draw=ImageDraw.Draw(image)
    def __getattr__(self,name):return getattr(self.draw,name)
    def text(self,xy,text,*args,**kwargs):
        text=unicodedata.normalize('NFKD',str(text)).encode('ascii','ignore').decode()
        return self.draw.text(xy,text,*args,**kwargs)

def font(size=20):
    # Fuente incorporada en Pillow: mismo aspecto sin depender de fuentes del equipo.
    return ImageFont.load_default(size=size)

def base(title, subtitle, height=440):
    im=Image.new('RGB',(1200,height),'#f7fafc'); d=PortableDraw(im)
    d.text((35,22),title,font=font(26),fill=DARK)
    d.text((35,62),subtitle,font=font(16),fill='#536773')
    return im,d

def bars(labels, values, title, subtitle, unit='imágenes', color=BLUE):
    """Barras desde cero. No confundir frecuencia de imágenes con frecuencia de píxeles."""
    im,d=base(title,subtitle,max(320,140+len(labels)*47))
    left=360; width=700; maxv=max(values,default=1) or 1
    for i,(label,value) in enumerate(zip(labels,values)):
        y=112+i*47
        d.text((35,y+3),str(label),font=font(17),fill=DARK)
        d.rounded_rectangle((left,y,left+max(1,width*value/maxv),y+27),radius=4,fill=color)
        formatted=f'{value:,.0f}' if float(value).is_integer() else f'{value:.2f}'
        d.text((left+width*value/maxv+10,y+3),formatted,font=font(17),fill=DARK)
    d.text((left,im.height-30),'0  |  '+unit+'; longitud proporcional al valor',font=font(15),fill='#536773')
    return im

def composition(labels, values, title, subtitle):
    im,d=base(title,subtitle,340); total=sum(values); x=35
    for label,value,color in zip(labels,values,[BLUE,ORANGE,'#a9b7c2']):
        width=1130*value/total
        if width:d.rectangle((x,125,x+width,195),fill=color)
        x+=width
    for i,(label,value,color) in enumerate(zip(labels,values,[BLUE,ORANGE,'#a9b7c2'])):
        x=35+i*390
        d.rectangle((x,230,x+20,250),fill=color)
        d.text((x+30,225),f'{label}: {value/total:.2%}',font=font(21),fill=DARK)
    return im

def histogram(values,title,xlabel,bins=12):
    """Cada barra cuenta imágenes en un intervalo; línea naranja = mediana."""
    values=np.asarray(values,dtype=float)
    counts,edges=np.histogram(values,bins=bins)
    im,d=base(title,f'n = {len(values):,} imágenes | mediana = {np.median(values):.2f}',480)
    left,top,w,h=80,125,1050,260; maximum=max(int(counts.max()),1)
    for tick in np.linspace(0,maximum,5):
        y=top+h-h*tick/maximum
        d.line((left,y,left+w,y),fill='#dce5eb')
        d.text((15,y-9),f'{tick:.0f}',font=font(15),fill=DARK)
    for i,n in enumerate(counts):
        x=left+i*w/len(counts); bh=h*n/maximum
        if n:d.rectangle((x+2,top+h-bh,x+w/len(counts)-2,top+h),fill=BLUE)
    medianx=left+w*(np.median(values)-edges[0])/(edges[-1]-edges[0])
    d.line((medianx,top,medianx,top+h),fill=ORANGE,width=3)
    for i in np.linspace(0,len(counts),5,dtype=int):
        d.text((left+i*w/len(counts)-15,top+h+12),f'{edges[i]:.2f}',font=font(16),fill=DARK)
    d.text((80,440),xlabel+' | eje vertical: número de imágenes',font=font(17),fill=DARK)
    return im

def scatter(x,y,title,xlabel,ylabel):
    x=np.asarray(x);y=np.asarray(y)
    im,d=base(title,f'n = {len(x):,} imágenes; cada punto representa una imagen',560)
    left,top,w,h=100,120,1030,330
    xmax=max(float(x.max()),1)*1.05;ymax=max(float(y.max()),1)*1.05
    for t in np.linspace(0,1,6):
        xx=left+w*t;yy=top+h-h*t
        d.line((xx,top,xx,top+h),fill='#dce5eb');d.line((left,yy,left+w,yy),fill='#dce5eb')
        d.text((xx-10,top+h+12),f'{xmax*t:.0f}',font=font(16),fill=DARK)
        d.text((30,yy-7),f'{ymax*t:.0f}',font=font(16),fill=DARK)
    for a,b in zip(x,y):
        xx=left+w*a/xmax;yy=top+h-h*b/ymax
        d.ellipse((xx-2,yy-2,xx+2,yy+2),fill=BLUE)
    d.text((100,510),xlabel+' | vertical: '+ylabel,font=font(18),fill=DARK)
    return im
