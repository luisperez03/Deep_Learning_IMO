"""Actualiza la fila 65 del checklist con la revisión humana de la taxonomía; deja 13 y 67 como actualizar_checklist_errores2.

  python scripts/actualizar_checklist_errores3.py    # después de los actualizar_checklist_*.py anteriores; basta este
                                                     # (no hace falta correr antes actualizar_checklist_errores2.py)

Mismo mecanismo que scripts/actualizar_checklist_revision.py: reemplaza estado y evidencia por ID en
docs/revision_entregable2/CHECKLIST_ENTREGA_E2.md, falla si una fila no existe y escribe una sola nota (la reescribe si
se vuelve a correr). Filas 13 y 67: el texto de actualizar_checklist_errores2.filas(). Fila 65: P con
resultados/errores_atajos_e2/REVISION_HUMANA.json (python scripts/errores_atajos_e2.py medir-revision --hoja
hoja_revision_errores.xlsx). La revisión fue una CONFIRMACIÓN GLOBAL del revisor tras mirar los 88 casos del PDF, no
una marca caso por caso; la evidencia lo dice.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import actualizar_checklist_errores2 as E2  # noqa: E402

CHECK = E2.CHECK
MARCA = E2.MARCA


def filas():
    base, _ = E2.filas()
    rev = E2.leer('REVISION_HUMANA.json')
    loc, ata = E2.leer('RESUMEN_LOCAL.json'), E2.leer('RESUMEN_ATAJOS.json')
    if rev['casos_revisados'] != 88:
        raise SystemExit(f"Se esperaban 88 casos revisados y hay {rev['casos_revisados']}")
    casos = [t['casos_error'] for t in loc['taxonomia'].values()]
    comp = loc['errores_compartidos']
    ac = [r['taxonomia_exacta']['acuerdo_con_sustituto_HD95'] for r in ata['modelos'].values()]
    f65 = ('P', f"22/11, exploratorio: taxonomía previa de dos ejes mutuamente excluyentes (forma A1–A6, exacta con "
                f"componentes predichas; condición B1–B7); censo de errores de validación (Dice < 0,80), "
                f"{min(casos)}–{max(casos)} casos imagen-semilla por red, frecuencias con IC95 por grupos y sensibilidad "
                f"0,70/0,90; acuerdo del sustituto HD95 con la forma exacta {E2.rango(ac)}; "
                f"{comp['dificiles_8_de_10']}/{comp['imagenes_val']} imágenes fallan en ≥ 8 redes. Revisión humana de 88 "
                f"casos del top 3 (muestra estratificada, PDF con imagen, anotación y predicción): confirmación global "
                f"del revisor, sin registro individual. No se estima acuerdo independiente ni IC desde la hoja "
                f"completada por script. Pendiente registrar decisiones caso por caso. {E2.RUTA}")
    return {'13': base['13'], '65': f65, '67': base['67']}


def main():
    nuevas = filas()
    nota = (f"\n{MARCA} Prueba contrafactual de atajos en Colab (L4, 50 corridas), taxonomía de errores, subgrupos y daño "
            f"potencial con protocolo previo, exploratorio (filas 13 y 67); revisión humana de la taxonomía en 88 casos, "
            f"por confirmación global del revisor (fila 65).\n")
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
