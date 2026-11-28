"""Reproduce el mismo oráculo sin cambiar código ni tolerancia, variando solo OpenCV."""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cv-dir')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    if args.cv_dir:
        sys.path.insert(0, args.cv_dir)
    e3 = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(e3 / 'tests'))
    import test_original_model as t
    old = json.loads(t.JSON_GEOMETRIA.read_text())
    shapes = [('rectangulos_alto32_por_aspecto', str(a), [t._rect(32, a, d) for d in t.SUB])
              for a in [2, 4]]
    shapes += [('rombos_por_lado_px', str(l), [t._rombo(l, d) for d in t.SUB])
               for l in [8, 10, 12, 24, 32]]
    rows = []
    for group, key, masks in shapes:
        hashes = [hashlib.sha256(m.tobytes()).hexdigest() for m in masks]
        for variant in ['dbnet', 'c2', 'dbnet_at']:
            actual = t._medio(variant, masks)
            reference = old[group][key][variant]['dice']
            rows.append(dict(grupo=group, caso=key, variante=variant, actual=actual,
                             referencia=reference, diferencia=actual-reference,
                             aprueba=abs(actual-reference) <= .005, mascaras_sha256=hashes))
    report = dict(opencv=t.cv2.__version__, numpy=t.np.__version__,
                  geometria_sha256=hashlib.sha256((e3 / 'src/geometria_at.py').read_bytes()).hexdigest(),
                  comparaciones=rows, todas_aprueban=all(r['aprueba'] for r in rows))
    Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(report['opencv'], sum(r['aprueba'] for r in rows), '/', len(rows), flush=True)


if __name__ == '__main__':
    main()
