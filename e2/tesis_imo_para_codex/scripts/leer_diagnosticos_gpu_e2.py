"""Tablas de lectura del complemento GPU (filas 57-60, 62 y 64 del checklist).

Lee resultados/diagnosticos_gpu_e2/RESUMEN_DIAGNOSTICOS.json (salida de resumir_diagnosticos_gpu_e2.py en Colab) y
resultados/auditoria_test_e2/AUDITORIA.json (Dice de test cerrado), y escribe TABLAS.md en la misma carpeta. No
recalcula nada: ordena y redondea lo que ya trae el resumen, y compara el Dice de test de la etapa clean con el
cerrado para comprobar que los checkpoints evaluados son los mismos.

  python scripts/leer_diagnosticos_gpu_e2.py
"""
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
CARPETA = PROJECT / 'resultados/diagnosticos_gpu_e2'
RESUMEN = CARPETA / 'RESUMEN_DIAGNOSTICOS.json'
AUDITORIA = PROJECT / 'resultados/auditoria_test_e2/AUDITORIA.json'


def leer(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def f(x, d=3):
    return f'{x:.{d}f}'.replace('.', ',')


def ic(par, d=3):
    return f'[{f(par[0], d)}; {f(par[1], d)}]'


def main():
    r = leer(RESUMEN)
    if r['status'] != 'complete' or r['missing']:
        raise SystemExit(f"Resumen incompleto: {r['status']} {r['missing']}")
    cerrado = {t['model']: t for t in leer(AUDITORIA)['table']}
    clean = sorted(r['clean'], key=lambda c: -c['test']['Dice']['mean'])
    out = ['# Complemento GPU: tablas de lectura', '',
           f"Generado por `scripts/leer_diagnosticos_gpu_e2.py` desde `RESUMEN_DIAGNOSTICOS.json` (status "
           f"`{r['status']}`). {r['scope']} Media de 5 semillas; IC95 por bootstrap de grupos con semillas fijas.", '']

    out += ['## Reproducción del test cerrado y brecha train/validación (fila 57)', '',
            '| Red | Dice test clean | Dice test cerrado | Dif. | Dice train | Dice val | Brecha train−val [IC95] |',
            '|---|---:|---:|---:|---:|---:|---:|']
    max_dif = 0.0
    for c in clean:
        m = c['model']; t = c['test']['Dice']['mean']; z = cerrado[m]['Dice_test']; max_dif = max(max_dif, abs(t - z))
        g = c['gap_train_val']
        out.append(f"| {m} | {f(t, 4)} | {f(z, 4)} | {f(t - z, 4)} | {f(c['train']['Dice']['mean'])} | "
                   f"{f(c['val']['Dice']['mean'])} | {f(g['difference'])} {ic(g['CI95'])} |")
    out += ['', f'Diferencia máxima con el test cerrado: {f(max_dif, 4)}.', '']

    out += ['## Calibración en test (fila 58)', '',
            '| Red | ECE15 bruto | ECE15 con temperatura | Brier bruto | Brier con temperatura | Temperatura (semillas) |',
            '|---|---:|---:|---:|---:|---|']
    for c in clean:
        t = c['test']
        temps = ', '.join(f(x, 2) for x in c['temperature'])
        out.append(f"| {c['model']} | {f(t['ECE15']['mean'], 4)} | {f(t['ECE15_calibrated']['mean'], 4)} | "
                   f"{f(t['Brier']['mean'], 4)} | {f(t['Brier_calibrated']['mean'], 4)} | {temps} |")

    out += ['', '## Bordes, precisión y recall en test (fila 64)', '',
            '| Red | HD95 (px) | ASSD (px) | Precisión | Recall | Predicción vacía |', '|---|---:|---:|---:|---:|---:|']
    for c in clean:
        t = c['test']
        out.append(f"| {c['model']} | {f(t['HD95']['mean'], 1)} | {f(t['ASSD']['mean'], 1)} | "
                   f"{f(t['precision']['mean'])} | {f(t['recall']['mean'])} | {f(t['empty_prediction']['mean'])} |")

    out += ['', '## Costo de inferencia', '', '| Red | Parámetros | GFLOPs | Latencia mediana (ms) |',
            '|---|---:|---:|---:|']
    for c in clean:
        s = c['resources']
        params = f"{int(s['parameters_total']):,}".replace(',', '.')
        out.append(f"| {c['model']} | {params} | {f(s['profile_counted_FLOPs'] / 1e9, 1)} | "
                   f"{f(s['latency_ms_median'], 1)} |")

    out += ['', '## Robustez en test (fila 59)', '',
            'Área bajo la curva de Dice en severidad ordinal 0–3, normalizada por el Dice limpio (1 = no cae).', '',
            '| Red | Ruido [IC95] | Desenfoque [IC95] | Brillo [IC95] |', '|---|---:|---:|---:|']
    rob = {}
    for x in r['robustness']:
        rob.setdefault(x['model'], {})[x['kind']] = x
    for c in clean:
        fila = rob[c['model']]
        out.append(f"| {c['model']} | " + ' | '.join(
            f"{f(fila[k]['normalized_area_ordinal'])} {ic(fila[k]['CI95_groups'])}" for k in ('noise', 'blur', 'brightness'))
            + ' |')

    out += ['', '## Ensamble de semillas y predicción conforme (fila 62)', '',
            '| Red | Dice ensamble [IC95] | Ganancia sobre la semilla media [IC95] | Cobertura conforme (nominal 0,90) | '
            'Grupos bajo el nominal | Tamaño medio del conjunto |', '|---|---:|---:|---:|---:|---:|']
    for e in r['ensembles']:
        mc = e['conformal']['mondrian_by_class']['all']
        out.append(f"| {e['model']} | {f(e['means']['Dice'])} {ic(e['Dice_CI95_groups'])} | "
                   f"{f(e['gain_over_mean_member']['difference'])} {ic(e['gain_over_mean_member']['CI95_groups'])} | "
                   f"{f(mc['coverage'])} | {f(mc['by_group']['fraction_below_nominal'], 2)} | {f(mc['mean_set_size'])} |")

    out += ['', '## Explicabilidad: Integrated Gradients en validación (fila 60)', '',
            'Ventaja de IG sobre un orden aleatorio al borrar (caída) o insertar (subida) el 10 % de píxeles más '
            'atribuidos, en unidades del logit medio de la región del rótulo: no es comparable entre redes, que tienen '
            'escalas de logit distintas. Estabilidad como Spearman medio entre pares de semillas.', '',
            '| Red | Borrado: ventaja [IC95] | Casos donde IG gana | Inserción: ventaja [IC95] | Spearman entre semillas [IC95] |',
            '|---|---:|---:|---:|---:|']
    for i in r['interpretability']:
        d = i['fidelity']['0.1']
        out.append(f"| {i['model']} | {f(d['deletion_advantage_over_random_mean'], 2)} {ic(d['deletion_CI95_cases'], 2)} | "
                   f"{f(d['deletion_fraction_cases_IG_better'], 2)} | {f(d['insertion_advantage_over_random_mean'], 2)} "
                   f"{ic(d['insertion_CI95_cases'], 2)} | {f(i['mean_pairwise_spearman'])} {ic(i['spearman_CI95_cases'])} |")
    (CARPETA / 'TABLAS.md').write_text('\n'.join(out) + '\n', encoding='utf-8')
    print(CARPETA / 'TABLAS.md')


if __name__ == '__main__':
    main()
