"""Coteja contra el PDF de arXiv las cifras de RESULTADOS_ORIGINALES_7REDES.md.

Uso: python docs/revision_entregable2/cotejar_resultados_originales.py <carpeta_pdfs>
  - Si un PDF falta en la carpeta, lo descarga de https://arxiv.org/pdf/<id>.
  - Extrae el texto con `pdftotext -enc UTF-8 -raw` (poppler) y separa paginas por salto de pagina.
  - Cada afirmacion es (red, version arXiv, pagina impresa en el PDF, fragmento literal).
    Se confirma si el fragmento aparece en esa pagina del texto extraido.
Salida: docs/revision_entregable2/cotejo_resultados_originales.json
Los PDF no se versionan (tamano y licencia); la version arXiv queda fijada aqui.
"""
import datetime as dt
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# (red, arxiv, pagina, fragmento literal, que respalda)
CLAIMS = [
    ('DeepLabv3+', '1802.02611v3', 13, 'DeepLabv3+ (Xception) 87.8', 'VOC 2012 test 87,8 (T. 6)'),
    ('DeepLabv3+', '1802.02611v3', 13, 'DeepLabv3+ (Xception-JFT) 89.0', 'VOC 2012 test 89,0 con JFT (T. 6)'),
    ('DeepLabv3+', '1802.02611v3', 14, 'DeepLabv3+ 82.1', 'Cityscapes test 82,1 (T. 7b)'),
    ('DeepLabv3+', '1802.02611v3', 12, '16 16 79.79% 54.17B', 'Costo: 54,17 B Multiply-Adds, VOC val (T. 5)'),
    ('UNet++', '1912.05074v2', 6, '9.0M 88.92±0.14 91.03±1.34 92.44±1.20 90.86±0.81 82.51±1.29', 'UNet++ sin DS, IoU media±DE (T. IV)'),
    ('UNet++', '1912.05074v2', 6, '7.8M 88.30±0.24 88.73±1.64 90.57±1.26 89.21±1.55 79.90±1.38', 'U-Net de referencia (T. IV)'),
    ('UNet++', '1912.05074v2', 6, 'for 20 independent trials', '20 corridas independientes y prueba t (T. IV)'),
    ('PIDNet-S', '2206.02066v3', 8, 'PIDNet-S 78.8 78.6 93.2 RTX 3090 2048×1024 47.6 7.6M', 'Cityscapes val/test, FPS, GFLOPs, parametros (T. 6)'),
    ('PIDNet-S', '2206.02066v3', 7, 'PIDNet-S† 80.1 153.7 RTX 3090', 'CamVid 80,1 con preentrenamiento en Cityscapes (T. 5)'),
    ('SegNeXt-T', '2209.08575v1', 9, 'SegNeXt-T 4.3 6.6 41.1 42.2 50.5 79.8 81.4 6.6 38.7 39.1', 'ADE20K, Cityscapes, COCO-Stuff (T. 9)'),
    ('SegNeXt-T', '2209.08575v1', 9, 'SegNeXt-T 768 × 1,536 78.0', 'Cityscapes tiempo real (T. 11)'),
    ('SegFormer-B0', '2105.15203v3', 7, 'MiT-B0 3.4 0.4 8.4 37.4 / 38.0 125.5 76.2 / 78.1 8.4 35.6', 'Parametros, GFLOPs y mIoU SS/MS (T. 1a)'),
    ('SegFormer-B0', '2105.15203v3', 7, '- - - 17.7 47.6 71.9', 'Cityscapes lado corto 512: 71,9 a 47,6 FPS (T. 2)'),
    ('TransUNet', '2102.04306v1', 6, 'TransUNet 77.48 31.69 87.23 63.13 81.87 77.02 94.08 55.86 85.08 75.62', 'Synapse DSC y HD95 (T. 1)'),
    ('TransUNet', '2102.04306v1', 10, 'TransUNet 89.71 88.86 84.53 95.73', 'ACDC DSC (T. 5)'),
    ('VM-UNet', '2402.02491v2', 5, 'VM-UNet 80.23 89.03 96.29 97.58 89.90', 'ISIC17 mIoU, DSC, Acc, Spe, Sen (T. I)'),
    ('VM-UNet', '2402.02491v2', 5, 'VM-UNet 81.35 89.71 94.91 96.13 91.12', 'ISIC18 (T. I)'),
    ('VM-UNet', '2402.02491v2', 6, 'VM-UNet 81.08 19.21 86.40 69.41 86.16 82.76 94.17 58.80 89.51 81.40', 'Synapse DSC y HD95 (T. II)'),
    ('VM-UNet', '2402.02491v2', 6, '{2,2,2,2-2,2,2,1} 27.43M 4.11G 80.23 89.03 81.35 89.71', 'Configuracion base: parametros y FLOPs (T. V)'),
    ('VM-UNet', '2402.02491v2', 6, '- 77.59 87.38 78.66 88.06', 'Sin preentrenamiento (T. III)'),
    ('VM-UNet', '2402.02491v2', 6, '{2,2,9,2-2,9,2,2} 44.27M 7.56G 79.67 88.68 81.05 89.54', 'Variante implementada en el benchmark (T. V)'),
]


def texto(pdf):
    out = subprocess.run(['pdftotext', '-enc', 'UTF-8', '-raw', str(pdf), '-'], capture_output=True, check=True)
    return out.stdout.decode('utf-8', errors='replace').split('\f')


def main(carpeta):
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    paginas, resultado = {}, []
    for red, arxiv, pagina, frag, respalda in CLAIMS:
        if arxiv not in paginas:
            pdf = carpeta / f'{arxiv}.pdf'
            if not pdf.exists():
                req = urllib.request.Request(f'https://arxiv.org/pdf/{arxiv}', headers={'User-Agent': 'cotejo/1.0'})
                pdf.write_bytes(urllib.request.urlopen(req, timeout=60).read())
            paginas[arxiv] = texto(pdf)
        ok = frag in paginas[arxiv][pagina - 1]
        resultado.append({'red': red, 'arxiv': arxiv, 'pagina': pagina, 'fragmento': frag,
                          'respalda': respalda, 'confirmado': ok})
        print(('OK   ' if ok else 'FALLA'), red, arxiv, f'p{pagina}', respalda)
    resumen = {'fecha': dt.date.today().isoformat(), 'confirmadas': sum(r['confirmado'] for r in resultado),
               'total': len(resultado), 'afirmaciones': resultado}
    (ROOT / 'cotejo_resultados_originales.json').write_text(json.dumps(resumen, ensure_ascii=False, indent=2),
                                                           encoding='utf-8')
    print(f"{resumen['confirmadas']} de {resumen['total']} confirmadas")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('Indique una carpeta FUERA del repositorio para los PDF (no se versionan).')
    main(sys.argv[1])
