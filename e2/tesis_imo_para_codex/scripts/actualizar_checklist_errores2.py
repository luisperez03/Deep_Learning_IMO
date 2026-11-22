"""Actualiza el checklist maestro con errores, subgrupos, daño y atajos (filas 13, 65 y 67), tras la corrida en Colab.

  python scripts/actualizar_checklist_errores2.py    # después de los actualizar_checklist_*.py anteriores

Mismo mecanismo que scripts/actualizar_checklist_revision.py: reemplaza estado y evidencia por ID en
docs/revision_entregable2/CHECKLIST_ENTREGA_E2.md y falla si una fila no existe; la nota se agrega una sola vez y se
reescribe si se vuelve a correr. Lee la evidencia ejecutada de resultados/errores_atajos_e2/:
  RESUMEN_LOCAL.json    taxonomía automática, subgrupos y omisión grave (local)
  RESUMEN_ATAJOS.json   prueba contrafactual de atajos, negativos sintéticos y taxonomía exacta (Colab L4, 50/50)
  REVISION_HUMANA.json  acuerdo humano con la taxonomía (88 casos del top 3); mientras no exista, la 65 queda en P
Las filas 13 y 67 quedan en C: la prueba de atajos y los negativos están ejecutados con reglas fijadas antes.
"""
import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
CHECK = RAIZ / 'docs' / 'revision_entregable2' / 'CHECKLIST_ENTREGA_E2.md'
RES = RAIZ / 'resultados' / 'errores_atajos_e2'
RUTA = ('`resultados/errores_atajos_e2/` (`LECTURA.md`, `TABLAS_ATAJOS.md`), protocolo previo del 22/11 '
        '(`PROTOCOLO_ERRORES_ATAJOS_20261122.md`), `colab/Atajos_Errores_E2.ipynb`')
MARCA = '**Actualización del 22/11 (errores y atajos).**'


def leer(nombre, obligatorio=True):
    p = RES / nombre
    if not p.exists():
        if obligatorio:
            raise SystemExit(f'Falta {p}')
        return None
    return json.loads(p.read_text(encoding='utf-8'))


def c2(x):
    return f'{x:.2f}'.replace('.', ',')


def rango(xs):
    return f'{c2(min(xs))}–{c2(max(xs))}'


def filas():
    loc, ata, rev = leer('RESUMEN_LOCAL.json'), leer('RESUMEN_ATAJOS.json'), leer('REVISION_HUMANA.json', False)
    mod = ata['modelos']
    if len(mod) != 10 or any(len(r['semillas']) != 5 for r in mod.values()):
        raise SystemExit('RESUMEN_ATAJOS.json no tiene las 10 redes x 5 semillas')
    nombre = {'transunet_2d': 'TransUNet', 'fcn': 'FCN', 'unet': 'U-Net', 'vmunet': 'VM-UNet', 'segnet': 'SegNet',
              'deeplabv3plus': 'DeepLabv3+', 'unetplusplus': 'U-Net++', 'pidnet_s': 'PIDNet-S', 'segnext_t': 'SegNeXt-T',
              'segformer_b0': 'SegFormer-B0'}
    dec = lambda c, d: [nombre[m] for m, r in mod.items() if r[c]['decision'] == d]
    ctx, ctx_ind = dec('C1', 'dependencia material'), dec('C1', 'indeterminado')
    col = dec('C3', 'dependencia material')
    c2_si = [m for m, r in mod.items() if r['C2_decision'] == 'atajo de contexto']
    zona = [r['C2_fraccion_zona_predicha']['estimacion'] for r in mod.values()]
    fa = [r['C2_tasa_falsa_alarma']['estimacion'] for r in mod.values()]
    fa0 = [r['C2_falsa_alarma_por_exceso_previo']['sin_exceso_distante_en_limpia']['tasa'] for r in mod.values()]
    d3 = [r['C3']['delta'] for r in mod.values()]
    f13 = ('C', f"22/11, exploratorio, reglas previas (δ = 0,02, Holm): prueba contrafactual en validación, 10 redes × 5 "
                f"semillas (Colab L4). Sin contexto: dependencia material en {', '.join(ctx)}; indeterminado en "
                f"{', '.join(ctx_ind)}; sin dependencia en las otras {10 - len(ctx) - len(ctx_ind)}. Rótulo borrado: "
                f"zona predicha {rango(zona)}, ninguna red lo infiere del contexto. Gris: caída {rango([-x for x in d3])} en "
                f"las 10 (el color es parte del rótulo, no un atajo en sí). Señal del EDA (origen de captura): O1–O6 sin "
                f"diferencias. Imágenes sintéticas fuera de distribución, declarado. {RUTA}")
    f67 = ('C', f"22/11, exploratorio: subgrupos (clase, área, borde, n.º de rótulos, luz, contraste, origen) con IC95 por "
                f"grupos y Holm; caen el cuartil de área menor (6 de 10 redes) y el tercil de poca luz (7 de 10). Daño: "
                f"omisión grave (recall < 0,5) 3,8–7,7 % en nueve redes y 20 % en SegFormer, 8–12 % con poca luz en el "
                f"top 3. Negativos sintéticos (rótulos borrados): falsa alarma {rango(fa)} ({rango(fa0)} sin exceso previo "
                f"en la imagen limpia); sin negativos reales ni validación en campo, declarado. Atajos: fila 13. {RUTA}")
    tax = loc['taxonomia']
    casos = [t['casos_error'] for t in tax.values()]
    comp = loc['errores_compartidos']
    ac = [r['taxonomia_exacta']['acuerdo_con_sustituto_HD95'] for r in mod.values()]
    base65 = (f"22/11, exploratorio: taxonomía previa de dos ejes mutuamente excluyentes (forma A1–A6, exacta con "
              f"componentes predichas; condición B1–B7); censo de errores de validación (Dice < 0,80), "
              f"{min(casos)}–{max(casos)} casos imagen-semilla por red, frecuencias con IC95 por grupos y sensibilidad "
              f"0,70/0,90; acuerdo del sustituto HD95 con la forma exacta {rango(ac)}; "
              f"{comp['dificiles_8_de_10']}/{comp['imagenes_val']} imágenes fallan en ≥ 8 redes")
    if rev:
        a, b = rev['acuerdo_A'], rev['acuerdo_B']
        f65 = ('C', base65 + f"; revisión humana de {rev['casos_revisados']} casos del top 3: acuerdo forma {c2(a['tasa'])} "
                             f"(IC95 Wilson {c2(a['IC95_wilson'][0])}–{c2(a['IC95_wilson'][1])}), condición {c2(b['tasa'])} "
                             f"({c2(b['IC95_wilson'][0])}–{c2(b['IC95_wilson'][1])}). {RUTA}")
    else:
        f65 = ('P', base65 + f". Etiquetado automático; falta la revisión humana de 88 casos (hoja preparada). {RUTA}")
    return {'13': f13, '65': f65, '67': f67}, rev is not None


def main():
    nuevas, con_revision = filas()
    nota = (f"\n{MARCA} Prueba contrafactual de atajos en Colab (L4, 50 corridas), taxonomía de errores, subgrupos y daño "
            f"potencial con protocolo previo, exploratorio (filas 13 y 67)"
            + (" y revisión humana de la taxonomía (fila 65).\n" if con_revision else "; la fila 65 espera la revisión humana.\n"))
    lineas = CHECK.read_text(encoding='utf-8').splitlines()
    pendientes = dict(nuevas)
    for k, l in enumerate(lineas):
        celdas = l.split(' | ')
        if l.startswith('| ') and len(celdas) >= 4:
            ident = celdas[0][2:].strip()
            if ident in pendientes:
                est, ev = pendientes.pop(ident)
                lineas[k] = f'| {ident} | {celdas[1]} | {est} | {ev} |'
    if pendientes:
        raise SystemExit(f'Filas no encontradas: {sorted(pendientes)}')
    texto = '\n'.join(lineas) + '\n'
    if MARCA in texto:
        i = texto.index(MARCA); texto = texto[:i] + nota.strip() + texto[texto.index('\n', i):]
    else:
        texto = texto.replace('\n## Problema, hipótesis', nota + '\n## Problema, hipótesis', 1)
    CHECK.write_text(texto, encoding='utf-8')
    print({k: v[0] for k, v in nuevas.items()}, '->', CHECK)


if __name__ == '__main__':
    main()
