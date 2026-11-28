"""Reconstruye el corte completo en una carpeta NUEVA, comprobando todos los SHA-256."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifiesto',type=Path,default=Path(__file__).with_name('manifiesto_reconstruccion.json'))
    p.add_argument('--base',type=Path,required=True,help='ZIP anterior o carpeta que contiene tesis_imo_e3_secuencia_v4')
    p.add_argument('--actualizacion',type=Path,required=True)
    p.add_argument('--salida',type=Path,required=True,help='Carpeta nueva; no sobrescribe una campaña existente')
    a=p.parse_args();m=json.loads(a.manifiesto.read_text(encoding='utf-8'))
    if sha(a.actualizacion)!=m['actualizacion']['sha256']:raise ValueError('SHA-256 de actualización incorrecto')
    if a.salida.exists():raise ValueError('La salida debe ser nueva para preservar corridas existentes')
    old=None
    if a.base.is_file():
        if sha(a.base)!=m['base']['sha256']:raise ValueError('SHA-256 del ZIP base incorrecto')
        old=zipfile.ZipFile(a.base)
    elif not a.base.is_dir():raise FileNotFoundError(a.base)
    root=a.salida.resolve()
    # Primero verifica todos los contenidos; solo después escribe el nuevo corte.
    with zipfile.ZipFile(a.actualizacion) as delta:
        def read(record):
            rel=Path(record['ruta'])
            if rel.is_absolute() or '..' in rel.parts:raise ValueError('Ruta fuera del paquete')
            if record['origen']=='actualizacion':return delta.read(record['ruta'])
            if record['origen']!='base':raise ValueError('Origen desconocido')
            if old:return old.read(record['ruta'])
            source=(a.base/rel).resolve()
            if not source.is_relative_to(a.base.resolve()):raise ValueError('Ruta fuera de la base')
            return source.read_bytes()
        for record in m['archivos']:
            data=read(record)
            if len(data)!=record['bytes'] or hashlib.sha256(data).hexdigest()!=record['sha256']:
                raise ValueError('Contenido distinto: '+record['ruta'])
        for record in m['archivos']:
            target=(root/record['ruta']).resolve()
            if not target.is_relative_to(root):raise ValueError('Ruta fuera de salida')
            data=read(record)
            if hashlib.sha256(data).hexdigest()!=record['sha256']:raise ValueError('La base cambió durante la reconstrucción')
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    if old:old.close()
    print('Corte reconstruido y verificado:',root,'| archivos:',len(m['archivos']))


if __name__=='__main__':main()
