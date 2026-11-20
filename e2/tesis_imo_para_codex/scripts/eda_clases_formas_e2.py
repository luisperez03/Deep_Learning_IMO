"""EDA por clase: entropía de clases y de áreas, formas e histogramas de tono por categoría (checklist 09 y 16).

Completa lo que no estaba en resultados/complementos_locales_e2/ (allí: fracción de primer plano, entropía binaria
por imagen, regiones, perímetro e histogramas RGB primer plano/fondo por partición). Aquí:

1. Entropía de la distribución de clases (13 categorías): por imagen con presencia de la clase y por píxeles, en
   bits, normalizada por log2(13) y como número efectivo de clases (2^H). Divergencia de Jensen-Shannon train/val y
   train/test entre distribuciones de presencia (los conteos por clase de test ya estaban publicados en el libro).
2. Entropía de la distribución de áreas: fracción de primer plano por imagen en 12 bins logarítmicos fijos
   (1e-3 a 1), por partición, a partir de eda_por_imagen.json (no reabre el test).
3. Formas por clase (train y validación; test excluido): por región conexa de cada color de la máscara original,
   área relativa a la foto (píxeles), aspecto del rectángulo mínimo rotado, solidez (área del contorno/casco
   convexo), extensión (píxeles/caja alineada) y circularidad 4πA/P² con el área del contorno. Un rombo completo y de
   frente tiene solidez ~1, extensión ~0,5 y circularidad π/4 ≈ 0,785; regiones de menos de 30 píxeles se omiten.
4. Histograma de tono (HSV, 18 bins de 20°, píxeles con saturación > 0,25) de los píxeles de cada clase en train.

  python scripts/eda_clases_formas_e2.py --voc RUTA/VOC2012

Escribe resultados/eda_clases_e2/RESUMEN.json y tres figuras en libro/figuras/e2/eda_clases_*.png.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inicio_voc import palette  # noqa: E402
from preparar_base import PROJECT, read_csv  # noqa: E402

OUT = PROJECT / 'resultados/eda_clases_e2'
FIG = PROJECT / 'libro/figuras/e2'
BINS_AREA = np.logspace(-3, 0, 13)
ES = {'explosive': 'Explosivo', 'oxygen': 'Oxígeno', 'radioactive': 'Radiactivo', 'inhalation-hazard': 'Peligro por inhalación',
      'non-flammable-gas': 'Gas no inflamable', 'flammable-solid': 'Sólido inflamable', 'dangerous': 'Peligroso',
      'organic-peroxide': 'Peróxido orgánico', 'flammable': 'Inflamable', 'corrosive': 'Corrosivo',
      'infectious-substance': 'Sustancia infecciosa', 'spontaneously-combustible': 'Combustión espontánea', 'poison': 'Tóxico'}


def entropia(c):
    p = np.asarray(c, float); p = p[p > 0] / p.sum()
    return float(-(p * np.log2(p)).sum())


def jsd(a, b):
    a = np.asarray(a, float) / np.sum(a); b = np.asarray(b, float) / np.sum(b); m = (a + b) / 2
    return float((entropia_kl(a, m) + entropia_kl(b, m)) / 2)


def entropia_kl(p, q):
    s = p > 0
    return float((p[s] * np.log2(p[s] / q[s])).sum())


def resumen_entropia(conteos, k):
    h = entropia(conteos)
    return {'H_bits': h, 'H_normalizada': h / np.log2(k), 'clases_efectivas': 2 ** h}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--voc', type=Path, required=True)
    a = ap.parse_args(argv)
    colores = {c: n for c, n in palette(a.voc / 'labelmap.txt').items() if c != (0, 0, 0)}
    clases = sorted(colores.values())
    presencia = {s: dict.fromkeys(clases, 0) for s in ('train', 'val', 'test')}
    pixeles = {s: dict.fromkeys(clases, 0) for s in ('train', 'val')}
    formas = {s: {c: [] for c in clases} for s in ('train', 'val')}
    tono = {c: np.zeros(18, np.int64) for c in clases}
    for s in ('train', 'val', 'test'):
        for r in read_csv(PROJECT / 'resultados/preentrenamiento' / f'{s}.csv'):
            with Image.open(a.voc / r['original_mask_path'].split('VOC2012/', 1)[1]) as m:
                rgb = np.asarray(m.convert('RGB'))
            packed = rgb[..., 0].astype(np.int64) * 65536 + rgb[..., 1].astype(np.int64) * 256 + rgb[..., 2]
            presentes = {colores[c]: p for c in colores if (p := (packed == c[0] * 65536 + c[1] * 256 + c[2])).any()}
            for c in presentes:
                presencia[s][c] += 1
            if s == 'test':
                continue  # del test solo la presencia por clase, ya publicada en el libro
            total = packed.size
            hsv = None
            if s == 'train':
                with Image.open(a.voc / r['image_path'].split('VOC2012/', 1)[1]) as im:
                    hsv = cv2.cvtColor(np.asarray(im.convert('RGB')), cv2.COLOR_RGB2HSV)
            for c, mk in presentes.items():
                pixeles[s][c] += int(mk.sum())
                if hsv is not None:
                    sel = mk & (hsv[..., 1] > 64)
                    tono[c] += np.bincount((hsv[..., 0][sel].astype(int) * 2 // 20).clip(0, 17), minlength=18)
                n, lab, st, _ = cv2.connectedComponentsWithStats(mk.astype(np.uint8), connectivity=8)
                for k in range(1, n):
                    area = int(st[k, cv2.CC_STAT_AREA])
                    if area < 30:
                        continue
                    reg = (lab == k).astype(np.uint8)
                    cnt = max(cv2.findContours(reg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
                    per = cv2.arcLength(cnt, True); hull = cv2.contourArea(cv2.convexHull(cnt)); ac = cv2.contourArea(cnt)
                    (_, _), (w, h), _ = cv2.minAreaRect(cnt)
                    formas[s][c].append({'area_rel': area / total,
                                         'aspecto_rotado': float(max(w, h) / max(1e-6, min(w, h))),
                                         'solidez': float(ac / hull) if hull else 1.0,
                                         'extension': area / float(st[k, cv2.CC_STAT_WIDTH] * st[k, cv2.CC_STAT_HEIGHT]),
                                         'circularidad': float(4 * np.pi * ac / per ** 2) if per else 0.0,
                                         'toca_borde': bool(reg[0].any() or reg[-1].any() or reg[:, 0].any() or reg[:, -1].any())})
        print(s, 'listo', flush=True)

    eda = json.loads((PROJECT / 'resultados/complementos_locales_e2/eda_por_imagen.json').read_text(encoding='utf-8'))
    areas = {s: np.histogram(np.clip([r['foreground_fraction'] for r in eda if r['split'] == s], 1e-3, 1), BINS_AREA)[0]
             for s in ('train', 'val', 'test')}
    R = {'estado': 'Exploratorio, descriptivo. Formas e histogramas de tono solo en train/val; del test solo presencia por clase.',
         'entropia_clases_presencia': {s: {**resumen_entropia(list(presencia[s].values()), len(clases)), 'conteos': presencia[s]}
                                       for s in presencia},
         'entropia_clases_pixeles': {s: {**resumen_entropia(list(pixeles[s].values()), len(clases)), 'pixeles': pixeles[s]}
                                     for s in pixeles},
         'jensen_shannon_presencia_bits': {'train_val': jsd(list(presencia['train'].values()), list(presencia['val'].values())),
                                           'train_test': jsd(list(presencia['train'].values()), list(presencia['test'].values()))},
         'entropia_areas': {'bins_fraccion_primer_plano': BINS_AREA.round(5).tolist(),
                            **{s: {'conteos': areas[s].tolist(), 'H_bits': entropia(areas[s]),
                                   'H_normalizada': entropia(areas[s]) / np.log2(len(BINS_AREA) - 1)} for s in areas}},
         'formas_por_clase_train': {}, 'formas_por_clase_val_n': {c: len(formas['val'][c]) for c in clases},
         'tono_por_clase_train': {c: (tono[c] / max(1, tono[c].sum())).round(4).tolist() for c in clases},
         'tono_bins_grados': [f'{20 * k}-{20 * k + 20}' for k in range(18)]}
    for c in clases:
        f = formas['train'][c]
        R['formas_por_clase_train'][c] = {'regiones': len(f), **{
            k: {'mediana': float(np.median([x[k] for x in f])), 'p10': float(np.percentile([x[k] for x in f], 10)),
                'p90': float(np.percentile([x[k] for x in f], 90))}
            for k in ('area_rel', 'aspecto_rotado', 'solidez', 'extension', 'circularidad')},
            'fraccion_toca_borde': float(np.mean([x['toca_borde'] for x in f]))}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'RESUMEN.json').write_text(json.dumps(R, indent=1, ensure_ascii=False), encoding='utf-8')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    orden = sorted(clases, key=lambda c: np.median([x['area_rel'] for x in formas['train'][c]]))
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.boxplot([[100 * x['area_rel'] for x in formas['train'][c]] for c in orden], vert=False, showfliers=False)
    ax.set_xscale('log'); ax.set_yticks(range(1, len(orden) + 1), [ES[c] for c in orden])
    ax.set_xlabel('Área de cada región, % de la foto (escala log)'); ax.set_title('Tamaño de los rótulos por clase, train')
    fig.tight_layout(); fig.savefig(FIG / 'eda_clases_areas.png', dpi=130); plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(12, 5.5), sharey=True)
    for ax, k, t in zip(axs, ('solidez', 'extension', 'circularidad'), ('Solidez', 'Extensión', 'Circularidad')):
        ax.boxplot([[x[k] for x in formas['train'][c]] for c in orden], vert=False, showfliers=False)
        ax.set_title(t); ax.set_xlim(0, 1.05)
    axs[0].set_yticks(range(1, len(orden) + 1), [ES[c] for c in orden])
    fig.suptitle('Forma de las regiones por clase, train'); fig.tight_layout(); fig.savefig(FIG / 'eda_clases_formas.png', dpi=130)
    plt.close(fig)
    M = np.array([R['tono_por_clase_train'][c] for c in orden])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    im = ax.imshow(M, aspect='auto', cmap='viridis')
    ax.set_yticks(range(len(orden)), [ES[c] for c in orden]); ax.set_xticks(range(0, 18, 2), [f'{20 * k}°' for k in range(0, 18, 2)])
    ax.set_xlabel('Tono (HSV), píxeles con saturación > 0,25'); ax.set_title('Histograma de tono de los píxeles de cada clase, train')
    fig.colorbar(im, ax=ax, label='fracción de píxeles'); fig.tight_layout(); fig.savefig(FIG / 'eda_clases_tono.png', dpi=130)
    plt.close(fig)
    print(json.dumps({k: R[k] for k in ('entropia_areas', 'jensen_shannon_presencia_bits')}, indent=1, ensure_ascii=False)[:1500])
    print({s: round(R['entropia_clases_presencia'][s]['H_normalizada'], 4) for s in presencia})


if __name__ == '__main__':
    main()
