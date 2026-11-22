"""Precribado automático por reglas explícitas, reconstruido sobre los 6.322 grupos conciliados.

Uso: python docs/revision_entregable2/precribado_reconstruido.py
Entrada: registros_conciliados.json (ignorado por git). Salidas: precribado_registros.json (ID -> categoría, sin
resúmenes; se versiona) y precribado_resumen.json (conteos, comparación con resumen_cribado.json).

Por qué existe: el precribado original (resumen_cribado.json, 6.325 grupos) solo dejó conteos; ni su script ni sus
decisiones por registro se conservaron. Este lo sustituye con las mismas ocho categorías y reglas a la vista. No es
el mismo filtro y sus conteos difieren; su error se mide con la muestra manual de cribado_nucleo.py, que es lo que
le da validez, no el parecido con el original.

Texto evaluado: título + resumen + palabras clave (KW, DE, ID). Las reglas se aplican en este orden, y gana la primera:
 1. EXCLUIR_TIPO: revisión de congreso, errata, editorial, carta, publicación retractada, o resumen del tipo
    "the proceedings contain N papers".
 2. RETENER_DOMINIO: rótulos o mercancía peligrosa (DOMINIO).
 3. Sin la raíz "segment": EXCLUIR_DETECCION si trata de detección o cajas; si no, EXCLUIR_TAREA.
 4. RETENER_ARQUITECTURA: el TÍTULO nombra una arquitectura candidata o de la misma línea.
 5. EXCLUIR_DOMINIO: medicina, biología, audio u otra modalidad no transferible a fotografía RGB 2D
    (sin arquitectura candidata en el título, por la regla 4).
 6. RETENER_MECANISMO: mecanismo transferible (objetos pequeños, bordes, multiescala, dilatación, atención,
    Transformer, espacio de estados, eficiencia).
 7. DUDA: el resto.
"""
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent

TIPO_EXCLUIDO = re.compile(r'conference[ _]review|erratum|editorial|letter|retract|note\b', re.I)
PROCEEDINGS = re.compile(r'proceedings contain \d+ papers', re.I)
DOMINIO = re.compile(r'hazmat|hazardous material|dangerous good|placard|\bimdg\b|\bghs\b|hazard (?:label|sign|symbol)|'
                     r'warning (?:label|sign)|\bun number\b', re.I)
DETECCION = re.compile(r'detect|bounding box|yolo|faster r-cnn|object recogni', re.I)
ARQUITECTURA = re.compile(r'segformer|deeplab|segnet|transunet|unet\s?\+\+|u-net\s?\+\+|pidnet|segnext|vm-?unet|'
                          r'swin-?unet|attention u-?net|fully convolutional|\bfcn\b|mask2former|maskformer|setr\b|'
                          r'hrnet|bisenet|ddrnet|vision mamba|mamba-?unet', re.I)
NO_TRANSFERIBLE = re.compile(r'medical|clinic|patient|\bmri\b|\bct\b|computed tomograph|tumou?r|lesion|cancer|'
                             r'histolog|patholog|cell\b|cells\b|nucle|retina|fundus|cardiac|organ|brain|lung|liver|'
                             r'polyp|skin lesion|dermoscop|ultrasound|x-ray|dental|biolog|microscop|protein|'
                             r'audio|speech|acoustic|eeg|ecg|point cloud|lidar|hyperspectral|\bsar\b', re.I)
MECANISMO = re.compile(r'small object|tiny|boundar|edge|contour|multi-?scale|dilat|atrous|attention|transformer|'
                       r'state space|mamba|lightweight|real-time|efficient', re.I)


def clasificar(g):
    r = g['registros'][0]
    m = r.get('metadatos_originales', {})
    tipos = ' '.join([r.get('tipo', '')] + m.get('M3', []) + m.get('DT', []))
    kw = ' '.join(m.get('KW', []) + m.get('DE', []) + m.get('ID', []))
    texto = f"{g['titulo']} {r.get('resumen', '')} {kw}"
    if TIPO_EXCLUIDO.search(tipos) or PROCEEDINGS.search(r.get('resumen', '')):
        return 'EXCLUIR_TIPO'
    if DOMINIO.search(texto):
        return 'RETENER_DOMINIO'
    if not re.search(r'segment', texto, re.I):
        return 'EXCLUIR_DETECCION' if DETECCION.search(texto) else 'EXCLUIR_TAREA'
    if ARQUITECTURA.search(g['titulo']):
        return 'RETENER_ARQUITECTURA'
    if NO_TRANSFERIBLE.search(texto):
        return 'EXCLUIR_DOMINIO'
    if MECANISMO.search(texto):
        return 'RETENER_MECANISMO'
    return 'DUDA'


def main():
    grupos = json.loads((ROOT / 'registros_conciliados.json').read_text(encoding='utf-8'))
    dec = {g['id']: clasificar(g) for g in grupos}
    original = json.loads((ROOT / 'resumen_cribado.json').read_text(encoding='utf-8'))['conteos_por_decision']
    conteo = Counter(dec.values())
    resumen = {'grupos': len(grupos), 'conteos': dict(sorted(conteo.items())),
               'original_6325_sin_registros': original,
               'nota': 'Reglas explícitas en precribado_reconstruido.py; no reproduce el filtro original, del que solo '
                       'quedan conteos. Error medido con la muestra manual (cribado_nucleo.py).'}
    (ROOT / 'precribado_registros.json').write_text(json.dumps(dec, ensure_ascii=False, indent=0), encoding='utf-8')
    (ROOT / 'precribado_resumen.json').write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding='utf-8')
    for k in sorted(set(conteo) | set(original)):
        print(f'{k:22s} reconstruido {conteo.get(k, 0):5d}   original {original.get(k, 0):5d}')


if __name__ == '__main__':
    main()
