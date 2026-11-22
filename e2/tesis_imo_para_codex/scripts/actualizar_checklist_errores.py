"""Actualiza el checklist maestro con taxonomía de errores, subgrupos, daño y atajos (filas 13, 65 y 67).

  python scripts/actualizar_checklist_errores.py    # después de los actualizar_checklist_*.py anteriores

Mismo mecanismo que scripts/actualizar_checklist_revision.py: reemplaza estado y evidencia por ID en
docs/revision_entregable2/CHECKLIST_ENTREGA_E2.md y falla si una fila no existe. La nota se agrega una sola vez.
El estado se deriva de la evidencia EJECUTADA que exista en resultados/errores_atajos_e2/:
  RESUMEN_LOCAL.json     taxonomía automática, subgrupos y omisión grave (local)        -> 13, 65, 67 quedan en P
  RESUMEN_ATAJOS.json    prueba contrafactual de atajos y falsas alarmas (Colab, GPU)   -> 13 y 67 pasan a C
  REVISION_HUMANA.json   acuerdo humano con la taxonomía en 88 casos del top 3          -> 65 pasa a C
Un plan o un cuaderno sin correr no cambia el estado.
"""
import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
CHECK = RAIZ / 'docs' / 'revision_entregable2' / 'CHECKLIST_ENTREGA_E2.md'
RES = RAIZ / 'resultados' / 'errores_atajos_e2'
RUTA = '`resultados/errores_atajos_e2/` (`LECTURA.md`), protocolo previo del 22/11 (`PROTOCOLO_ERRORES_ATAJOS_20261122.md`)'


def leer(nombre):
    p = RES / nombre
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else None


def c(x):
    return f'{x:.2f}'.replace('.', ',')


def filas():
    loc, ata, rev = leer('RESUMEN_LOCAL.json'), leer('RESUMEN_ATAJOS.json'), leer('REVISION_HUMANA.json')
    if loc is None:
        raise SystemExit('Falta resultados/errores_atajos_e2/RESUMEN_LOCAL.json: correr scripts/errores_atajos_e2.py analizar')
    tax = loc['taxonomia']
    casos = [t['casos_error'] for t in tax.values()]
    comp = loc['errores_compartidos']
    base65 = (f"22/11, exploratorio: taxonomía de dos ejes mutuamente excluyentes fijada antes (forma del error A1–A6; "
              f"condición de la imagen B1–B7); censo de errores de validación (Dice < 0,80), {min(casos)}–{max(casos)} "
              f"casos imagen-semilla por red, frecuencias con IC95 por grupos y sensibilidad 0,70/0,90; errores "
              f"compartidos: {comp['dificiles_8_de_10']}/{comp['imagenes_val']} imágenes fallan en ≥ 8 redes")
    if rev:
        a, b = rev['acuerdo_A'], rev['acuerdo_B']
        f65 = ('C', base65 + f"; revisión humana de {rev['casos_revisados']} casos del top 3: acuerdo A {c(a['tasa'])} "
                             f"(IC95 Wilson {c(a['IC95_wilson'][0])}–{c(a['IC95_wilson'][1])}), B {c(b['tasa'])} "
                             f"({c(b['IC95_wilson'][0])}–{c(b['IC95_wilson'][1])}). {RUTA}")
    else:
        f65 = ('P', base65 + f". Etiquetado automático; falta la revisión humana de 88 casos (`errores_atajos_e2.py "
                             f"revision`). {RUTA}")
    base67 = ("22/11, exploratorio: Dice por clase, área, borde, número de rótulos, luz, contraste y origen con IC95 "
              "por grupos y Holm (val; test descriptivo); caen el cuartil de área menor y el tercil de poca luz; omisión "
              "grave (recall < 0,5) 3,8–7,7 % en nueve redes y 20 % en SegFormer, mayor con poca luz")
    base13 = ("22/11, exploratorio: señal del EDA (origen de captura) contrastada por subconjuntos: O1–O6 sin diferencias; "
              "O7 (5 imágenes) mejor en las 10 redes")
    if ata:
        dec = {m: (r['C1']['decision'], r['C3']['decision'], r['C2_decision']) for m, r in ata['modelos'].items()}
        n_ctx = sum(d[0] == 'dependencia material' for d in dec.values())
        n_col = sum(d[1] == 'dependencia material' for d in dec.values())
        n_c2 = sum(d[2] == 'atajo de contexto' for d in dec.values())
        fa = [r['C2_tasa_falsa_alarma']['estimacion'] for r in ata['modelos'].values()]
        prueba = (f"; prueba contrafactual en validación, {len(dec)} redes × 5 semillas: dependencia material del contexto "
                  f"en {n_ctx}, del color en {n_col}, rótulo inferido del contexto en {n_c2}")
        f13 = ('C', base13 + prueba + f". {RUTA}")
        f67 = ('C', base67 + prueba + f"; falsa alarma en negativos sintéticos {c(min(fa))}–{c(max(fa))}. {RUTA}")
    else:
        f13 = ('P', base13 + f". Falta la prueba contrafactual (sin contexto, sin rótulo, gris): "
                             f"`colab/Atajos_Errores_E2.ipynb`, preparado sin correr. {RUTA}")
        f67 = ('P', base67 + f". Faltan negativos (sintéticos, C2) y atajos: `colab/Atajos_Errores_E2.ipynb`, preparado "
                             f"sin correr. {RUTA}")
    return {'13': f13, '65': f65, '67': f67}, ata is not None, rev is not None


def main():
    nuevas, con_atajos, con_revision = filas()
    hecho = ['taxonomía automática, subgrupos y daño (local)'] + (['atajos en Colab'] if con_atajos else []) + \
            (['revisión humana de la taxonomía'] if con_revision else [])
    hecho = ', '.join(hecho); hecho = hecho[0].upper() + hecho[1:]
    nota = (f"\n**Actualización del 22/11 (errores y atajos).** {hecho}: filas 13, 65 y 67, con "
            f"protocolo previo y exploratorio.\n")
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
    marca = '**Actualización del 22/11 (errores y atajos).**'
    if marca in texto:
        inicio = texto.index(marca); fin = texto.index('\n', inicio)
        texto = texto[:inicio] + nota.strip() + texto[fin:]
    else:
        texto = texto.replace('\n## Problema, hipótesis', nota + '\n## Problema, hipótesis', 1)
    CHECK.write_text(texto, encoding='utf-8')
    print({k: v[0] for k, v in nuevas.items()}, '->', CHECK)


if __name__ == '__main__':
    main()
