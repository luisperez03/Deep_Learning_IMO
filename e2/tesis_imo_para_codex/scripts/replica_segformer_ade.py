"""Réplica de evaluación de SegFormer-B0 en ADE20K (ítem 32 del checklist).

Sigue config/replica_segformer.json, fijado antes de ejecutar. Corre en Colab
(lo invoca colab/Replica_SegFormer_ADE20K.ipynb). Las funciones de
preprocesamiento, acumulación y bootstrap no dependen de transformers y se
prueban en tests/test_replica_segformer.py con un modelo simulado.

Uso en Colab:
  python replica_segformer_ade.py --ade-root /content/ade/ADEChallengeData2016 \
      --salida /content/replica_segformer --variante primaria
"""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

NUM_CLASES = 150
IGNORAR = 255
MEDIA_0_255 = np.array([123.675, 116.28, 103.53], dtype=np.float32)
DE_0_255 = np.array([58.395, 57.12, 57.375], dtype=np.float32)
MEDIA_IMAGENET = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DE_IMAGENET = np.array([0.229, 0.224, 0.225], dtype=np.float32)
HF_MODELO = 'nvidia/segformer-b0-finetuned-ade-512-512'
HF_REVISION = '489d5cd81a0b59fab9b7ea758d3548ebe99677da'
HF_SHA256 = '6ae39addd01de6b1b8bde2cf677d43a5cd733424b8d186de3f95d1c51fee23f9'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for bloque in iter(lambda: f.read(1 << 20), b''):
            h.update(bloque)
    return h.hexdigest()


# ---------------------------------------------------------------- preprocesamiento
def tamano_reescalado(h, w, escala=(2048, 512)):
    """mmcv.rescale_size con keep_ratio: factor min(largo_max/largo, corto_max/corto)."""
    largo_max, corto_max = max(escala), min(escala)
    factor = min(largo_max / max(h, w), corto_max / min(h, w))
    return int(h * factor + 0.5), int(w * factor + 0.5)


def tamano_alineado(h, w, divisor=32):
    return int(np.ceil(h / divisor)) * divisor, int(np.ceil(w / divisor)) * divisor


def preprocesar_primario(rgb):
    """AlignedResize de NVlabs/SegFormer: reescala con proporción y alinea a 32.

    Dos redimensiones bilineales con cv2 (como mmcv.imrescale y mmcv.imresize).
    Devuelve tensor CHW float32 normalizado y el tamaño de entrada.
    """
    import cv2
    h, w = rgb.shape[:2]
    nh, nw = tamano_reescalado(h, w)
    img = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    ah, aw = tamano_alineado(nh, nw)
    if (ah, aw) != (nh, nw):
        img = cv2.resize(img, (aw, ah), interpolation=cv2.INTER_LINEAR)
    img = (img.astype(np.float32) - MEDIA_0_255) / DE_0_255
    return np.ascontiguousarray(img.transpose(2, 0, 1)), (ah, aw)


def preprocesar_hf(rgb):
    """Variante D1: preprocesador por defecto de la ficha HF (512x512, sin proporción).

    SegformerImageProcessor: resize 512x512 con PIL bilineal (resample=2),
    reescala a 0-1 y normaliza con media/DE de ImageNet.
    """
    from PIL import Image
    img = Image.fromarray(rgb).resize((512, 512), resample=Image.BILINEAR)
    img = np.asarray(img, dtype=np.float32) / 255.0
    img = (img - MEDIA_IMAGENET) / DE_IMAGENET
    return np.ascontiguousarray(img.transpose(2, 0, 1)), (512, 512)


PREPROCESOS = {'primaria': preprocesar_primario, 'hf_512': preprocesar_hf}


def reducir_etiqueta_cero(etiqueta):
    """reduce_zero_label de mmseg: 0 -> 255, k -> k-1, y 255 se conserva."""
    e = etiqueta.astype(np.int64)
    salida = e - 1
    salida[e == 0] = IGNORAR
    salida[e == IGNORAR] = IGNORAR
    return salida


# ---------------------------------------------------------------- métrica
def interseccion_union(pred, gt, num_clases=NUM_CLASES, ignorar=IGNORAR):
    """Áreas por clase de una imagen, como intersect_and_union de mmseg 0.x."""
    valido = gt != ignorar
    p = pred[valido].astype(np.int64)
    g = gt[valido].astype(np.int64)
    inter = np.bincount(p[p == g], minlength=num_clases)[:num_clases]
    area_p = np.bincount(p, minlength=num_clases)[:num_clases]
    area_g = np.bincount(g, minlength=num_clases)[:num_clases]
    return inter, area_p + area_g - inter, area_g


def miou_acumulado(inter, union):
    """mIoU sobre el conjunto: suma por clase y nanmean de las clases con unión > 0."""
    inter = np.asarray(inter, dtype=np.float64)
    union = np.asarray(union, dtype=np.float64)
    if inter.ndim == 2:
        inter, union = inter.sum(0), union.sum(0)
    with np.errstate(invalid='ignore', divide='ignore'):
        iou = np.where(union > 0, inter / np.where(union > 0, union, 1), np.nan)
    return float(np.nanmean(iou)), iou


def miou_medio_por_imagen(inter, union):
    """Variante D2: media de imágenes del mIoU de cada imagen (no es la métrica del artículo)."""
    with np.errstate(invalid='ignore', divide='ignore'):
        iou = np.where(union > 0, inter / np.where(union > 0, union, 1), np.nan)
    return float(np.nanmean(np.nanmean(iou, axis=1)))


def bootstrap_miou(inter, union, replicas=2000, semilla=20261120, bloque=200):
    """IC percentil 95 % remuestreando imágenes; recalcula el mIoU acumulado."""
    rng = np.random.default_rng(semilla)
    n = inter.shape[0]
    inter = inter.astype(np.float64)
    union = union.astype(np.float64)
    valores = np.empty(replicas)
    hecho = 0
    while hecho < replicas:
        k = min(bloque, replicas - hecho)
        # pesos multinomiales = conteo de veces que entra cada imagen
        pesos = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(k)])
        si = pesos @ inter
        su = pesos @ union
        with np.errstate(invalid='ignore', divide='ignore'):
            iou = np.where(su > 0, si / np.where(su > 0, su, 1), np.nan)
        valores[hecho:hecho + k] = np.nanmean(iou, axis=1)
        hecho += k
    return valores


def veredicto(miou_pct, ic_pct, reportado=37.4, delta=1.0):
    d = miou_pct - reportado
    ic_d = [ic_pct[0] - reportado, ic_pct[1] - reportado]
    if abs(d) > delta:
        v = 'no_replicado'
    elif -delta <= ic_d[0] and ic_d[1] <= delta:
        v = 'replicado'
    else:
        v = 'compatible'
    return {'D_puntos': d, 'IC95_D': ic_d, 'veredicto': v,
            'reportado_dentro_IC95': bool(ic_pct[0] <= reportado <= ic_pct[1]),
            'delta': delta, 'reportado': reportado}


# ---------------------------------------------------------------- evaluación
def logits_a_prediccion(logits, tam_entrada, tam_original):
    """Reescala como mmseg: a la entrada y luego al tamaño original; argmax."""
    import torch.nn.functional as F
    x = F.interpolate(logits, size=tam_entrada, mode='bilinear', align_corners=False)
    x = F.interpolate(x, size=tam_original, mode='bilinear', align_corners=False)
    return x.argmax(1)[0].cpu().numpy()


def pares_validacion(ade_root):
    img_dir = Path(ade_root) / 'images' / 'validation'
    ann_dir = Path(ade_root) / 'annotations' / 'validation'
    imagenes = sorted(img_dir.glob('*.jpg'))
    pares = [(p, ann_dir / (p.stem + '.png')) for p in imagenes]
    faltan = [str(a) for _, a in pares if not a.exists()]
    if faltan:
        raise FileNotFoundError(f'Anotaciones faltantes: {faltan[:3]} (total {len(faltan)})')
    return pares


def evaluar(modelo_fn, pares, preprocesar, dispositivo, limite=None, avisar=200):
    """modelo_fn(tensor NCHW) -> logits NCHW (150 canales). Devuelve áreas por imagen."""
    import torch
    from PIL import Image
    pares = pares[:limite] if limite else pares
    inter = np.zeros((len(pares), NUM_CLASES), dtype=np.int64)
    union = np.zeros_like(inter)
    area_gt = np.zeros_like(inter)
    nombres = []
    t0 = time.perf_counter()
    for i, (ruta_img, ruta_ann) in enumerate(pares):
        rgb = np.asarray(Image.open(ruta_img).convert('RGB'))
        gt = reducir_etiqueta_cero(np.asarray(Image.open(ruta_ann)))
        if gt.shape != rgb.shape[:2]:
            raise ValueError(f'Tamaños distintos en {ruta_img.name}')
        x, tam = preprocesar(rgb)
        with torch.no_grad():
            logits = modelo_fn(torch.from_numpy(x)[None].to(dispositivo))
        pred = logits_a_prediccion(logits.float(), tam, gt.shape)
        inter[i], union[i], area_gt[i] = interseccion_union(pred, gt)
        nombres.append(ruta_img.stem)
        if avisar and (i + 1) % avisar == 0:
            print(f'{i + 1}/{len(pares)} imágenes, {time.perf_counter() - t0:.0f} s', flush=True)
    return {'nombres': nombres, 'inter': inter, 'union': union, 'area_gt': area_gt,
            'segundos': time.perf_counter() - t0}


def cargar_modelo_hf(dispositivo):
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import SegformerForSemanticSegmentation
    ruta = hf_hub_download(HF_MODELO, 'model.safetensors', revision=HF_REVISION)
    real = sha256(ruta)
    if real != HF_SHA256:
        raise RuntimeError(f'SHA-256 del checkpoint distinto: {real}')
    modelo = SegformerForSemanticSegmentation.from_pretrained(HF_MODELO, revision=HF_REVISION)
    modelo.eval().to(dispositivo)
    if modelo.config.num_labels != NUM_CLASES:
        raise RuntimeError('El checkpoint no tiene 150 clases')
    parametros = sum(p.numel() for p in modelo.parameters())
    return (lambda x: modelo(pixel_values=x).logits), {
        'modelo': HF_MODELO, 'revision': HF_REVISION, 'sha256_safetensors': real,
        'parametros': parametros, 'id2label': modelo.config.id2label}


def versiones():
    salida = {'python': platform.python_version()}
    for m in ['torch', 'transformers', 'huggingface_hub', 'numpy', 'cv2', 'PIL']:
        try:
            salida[m] = __import__(m).__version__
        except Exception as error:  # se registra la ausencia, no se oculta
            salida[m] = f'no disponible: {error}'
    try:
        import torch
        salida['gpu'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
    except Exception:
        salida['gpu'] = 'desconocida'
    return salida


def resumir(areas, protocolo, variante, info_modelo, nombres_clase=None):
    miou, iou_clase = miou_acumulado(areas['inter'], areas['union'])
    boot = bootstrap_miou(areas['inter'], areas['union'],
                          protocolo['incertidumbre']['replicas'], protocolo['incertidumbre']['semilla'])
    ic = [float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))]
    crit = protocolo['criterio_a_priori']
    resultado = {
        'variante': variante,
        'es_primaria': variante == 'primaria',
        'imagenes': len(areas['nombres']),
        'mIoU_pct': 100 * miou,
        'IC95_pct': [100 * ic[0], 100 * ic[1]],
        'bootstrap': {'replicas': len(boot), 'semilla': protocolo['incertidumbre']['semilla'],
                      'DE_pct': float(100 * np.nanstd(boot))},
        'mIoU_medio_por_imagen_pct_D2': 100 * miou_medio_por_imagen(areas['inter'], areas['union']),
        'IoU_por_clase_pct': {
            (nombres_clase.get(str(k), nombres_clase.get(k, str(k))) if nombres_clase else str(k)):
                (None if np.isnan(v) else float(100 * v)) for k, v in enumerate(iou_clase)},
        'segundos_inferencia': areas['segundos'],
        'modelo': {k: v for k, v in info_modelo.items() if k != 'id2label'},
        'versiones': versiones(),
    }
    resultado['criterio'] = veredicto(resultado['mIoU_pct'], resultado['IC95_pct'],
                                      protocolo['paper_guia']['valor_reportado_miou'],
                                      crit['delta_puntos_miou'])
    if not resultado['es_primaria']:
        resultado['criterio']['nota'] = 'Variante de análisis de desviación: no decide el veredicto.'
    return resultado


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ade-root', required=True)
    parser.add_argument('--salida', required=True)
    parser.add_argument('--protocolo', default=str(Path(__file__).resolve().parents[1] / 'config/replica_segformer.json'))
    parser.add_argument('--variante', choices=list(PREPROCESOS), default='primaria')
    parser.add_argument('--limite', type=int, help='Solo para medir tiempo; invalida el veredicto')
    args = parser.parse_args()
    protocolo = json.loads(Path(args.protocolo).read_text(encoding='utf-8'))
    import torch
    dispositivo = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pares = pares_validacion(args.ade_root)
    if len(pares) != 2000 and not args.limite:
        raise RuntimeError(f'Se esperaban 2.000 imágenes de validación, hay {len(pares)}')
    modelo_fn, info = cargar_modelo_hf(dispositivo)
    areas = evaluar(modelo_fn, pares, PREPROCESOS[args.variante], dispositivo, args.limite)
    resumen = resumir(areas, protocolo, args.variante, info, info.get('id2label'))
    resumen['sha256_protocolo'] = sha256(args.protocolo)
    if args.limite:
        resumen['criterio'] = {'veredicto': 'no_aplica', 'nota': f'Corrida parcial de {args.limite} imágenes para medir tiempo'}
    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(salida / f'areas_{args.variante}.npz', nombres=np.array(areas['nombres']),
                        inter=areas['inter'], union=areas['union'], area_gt=areas['area_gt'])
    (salida / f'resumen_{args.variante}.json').write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: resumen[k] for k in ['variante', 'imagenes', 'mIoU_pct', 'IC95_pct', 'criterio']},
                     indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
