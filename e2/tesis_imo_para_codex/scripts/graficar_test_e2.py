"""Figura del test auditado; no ejecuta inferencias."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1] / 'resultados/auditoria_test_e2'
report = json.loads((root / 'AUDITORIA.json').read_text(encoding='utf-8'))
assert not report['issues'], 'Resolver incidencias antes de graficar.'
rows = report['table']
names = {'transunet_2d':'TransUNet', 'fcn':'FCN', 'unet':'U-Net',
         'vmunet':'VM-UNet', 'segnet':'SegNet', 'deeplabv3plus':'DeepLabv3+',
         'unetplusplus':'U-Net++', 'pidnet_s':'PIDNet-S',
         'segnext_t':'SegNeXt-T', 'segformer_b0':'SegFormer-B0'}
fig, ax = plt.subplots(figsize=(10, 6.6))
for i, row in enumerate(rows):
    mean = row['Dice_test']
    lo, hi = row['CI95']
    ax.errorbar(mean, i, xerr=[[mean-lo], [hi-mean]], fmt='o',
                color='#176B87', capsize=4, label='Test: media e IC95' if i == 0 else None)
    ax.scatter(row['Dice_val'], i, marker='D', color='#D9822B', s=30,
               label='Validación: media' if i == 0 else None, zorder=3)
    ax.text(.912, i, f'{mean:.4f}', va='center', fontsize=10)
ax.set_yticks(range(len(rows)), [names[r['model']] for r in rows])
ax.invert_yaxis()
ax.set_xlim(.64, .96)
ax.set_xlabel('Dice: promedio con igual peso por grupo de similitud')
ax.set_title('E2 · Resultados de test en 10 redes y 5 semillas', loc='left', pad=16)
ax.grid(axis='x', alpha=.2)
ax.legend(loc='upper left', fontsize=9)
ax.spines[['top', 'right']].set_visible(False)
fig.text(.02, .025, '218 imágenes · 122 grupos · IC95: bootstrap de grupos, cinco semillas fijas.\n'
         'TransUNet − FCN: +0,91 pp; IC95 pareado [−0,83; +2,78] pp. No establece una mejora clara.', fontsize=9)
fig.tight_layout(rect=[0, .09, 1, 1])
fig.savefig(root / 'comparacion_validacion_test.png', dpi=180)
fig.savefig(root / 'comparacion_validacion_test.pdf')
plt.close(fig)
