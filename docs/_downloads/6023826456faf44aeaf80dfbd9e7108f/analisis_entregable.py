"""Reproduce las tablas del libro a partir de las evaluaciones cerradas."""
import csv
import hashlib
import io
import json
import tarfile
import zipfile
from collections import defaultdict
from itertools import combinations, permutations, product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / "libro"
MODELS = ["fcn", "unet", "segnet"]
LABEL = ["FCN", "U-Net", "SegNet"]
SEEDS = [11, 23, 37, 53, 71]
METRICS = ["Dice", "IoU", "mIoU", "accuracy"]


def fmt(x):
    return f"{x:.4f}".replace(".", ",")


def table(headers, rows):
    return "\n".join("| " + " | ".join(map(str, row)) + " |"
                     for row in [headers, ["---"] * len(headers), *rows])


def interval(values):
    return " a ".join(fmt(x) for x in np.quantile(values, [.025, .975]))


def signed_rank_exact(differences):
    """Distribución exacta de la suma de rangos bajo signos equiprobables."""
    d = differences[differences != 0]
    ranks = np.rint(2 * stats.rankdata(abs(d))).astype(int)
    counts = [1]
    for rank in ranks:
        expanded = counts + [0] * rank
        for total, count in enumerate(counts):
            expanded[total + rank] += count
        counts = expanded
    score = int(ranks[d > 0].sum())
    tail = min(score, int(ranks.sum())-score)
    return min(1., 2*sum(counts[:tail+1]) / (2**len(d)))


def csv_rows(data):
    return list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))


def main():
    assets = BOOK / "figuras"
    assets.mkdir(exist_ok=True)
    with zipfile.ZipFile(ROOT / "IMO_test_final.zip") as archive:
        detail = csv_rows(archive.read("por_imagen.csv"))
        runs = csv_rows(archive.read("por_corrida.csv"))
        summary = csv_rows(archive.read("resumen.csv"))
        meta = json.loads(archive.read("procedencia.json"))
    test = (ROOT / "resultados/preentrenamiento/test.csv").read_bytes()
    assert hashlib.sha256(test).hexdigest() == meta["frozen_files"]["test.csv"]
    mapping = {row["id"]: row["group_id"] for row in csv_rows(test)}
    groups = sorted(set(mapping.values()))
    assert len(detail) == 3270 and len(mapping) == 218 and len(groups) == 122
    assert meta["threshold"] == .5 and meta["lr"] == .001
    blocks = defaultdict(list)
    seen = set()
    for row in detail:
        key = (row["model"], int(row["seed"]), row["id"])
        assert key not in seen and mapping[key[2]] == row["group_id"]
        seen.add(key)
        blocks[key[:2] + (row["group_id"],)].append([float(row[k]) for k in METRICS])
    cube = np.array([[[np.mean(blocks[m, s, g], axis=0) for g in groups]
                      for s in SEEDS] for m in MODELS])
    assert cube.shape == (3, 5, 122, 4) and np.isfinite(cube).all()
    means = cube.mean(axis=2)
    assert len(runs) == 15 and len(summary) == 3
    for row in runs:
        np.testing.assert_allclose(means[MODELS.index(row["model"]), SEEDS.index(int(row["seed"]))],
                                   [float(row[k]) for k in METRICS])
    draws = np.random.default_rng(20261107).integers(122, size=(10000, 122))
    group = cube[:, :, :, 0].mean(axis=1)
    rows = []
    for i, m in enumerate(MODELS):
        r = next(r for r in summary if r["model"] == m)
        sims = group[i][draws].mean(axis=1)
        np.testing.assert_allclose([*means[i].mean(axis=0), means[i,:,0].std(ddof=1),
                                    *np.quantile(sims,[.025,.975])],
                                   [float(r[k]) for k in METRICS + ["Dice_SD_semillas",
                                   "Dice_IC95_grupos_inf","Dice_IC95_grupos_sup"]])
        rows.append([LABEL[i], *map(fmt, means[i].mean(axis=0)), fmt(means[i,:,0].std(ddof=1)), interval(sims)])
    sections = [table(["Red","Dice","IoU","mIoU","Exactitud","DE semillas","IC 95 % por grupos (Dice)"], rows)]
    comparisons = []
    bands = []
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3), constrained_layout=True)
    for ax, (i, j) in zip(axes, combinations(range(3), 2)):
        d = group[i] - group[j]
        ds = means[i,:,0] - means[j,:,0]
        contrast = LABEL[i] + " − " + LABEL[j]
        stats.probplot(ds, plot=ax)
        ax.set(title=contrast, xlabel="Cuantil normal", ylabel="Diferencia de Dice")
        p = stats.wilcoxon(ds, method="exact").pvalue
        samples = d[draws]
        low,high = np.quantile(samples.mean(axis=1),[.025,.975])
        bands.append((contrast,d.mean(),low,high))
        standardized = samples.mean(axis=1) / samples.std(axis=1, ddof=1)
        comparisons.append((p, [contrast, fmt(d.mean()), interval(samples.mean(axis=1)),
                                fmt(d.mean()/d.std(ddof=1)), interval(standardized),
                                fmt(np.mean(d > 0) + .5*np.mean(d == 0)), fmt(stats.shapiro(ds).pvalue), fmt(p)],
                            (stats.norm.ppf(1-.05/6)+stats.norm.ppf(.8))*ds.std(ddof=1)/np.sqrt(5)))
    fig.savefig(assets/"qq_semillas.png", dpi=150)
    plt.close(fig)
    fig,ax = plt.subplots(figsize=(8.5,3.3),constrained_layout=True)
    ax.axvline(0,color="#455869",lw=1.3,label="Diferencia cero")
    ax.axvline(.02,color="#a15a22",ls="--",lw=1.5,label="Margen práctico (+0,02)")
    for y,(_,delta,low,high) in enumerate(bands):
        ax.errorbar(delta,y,xerr=[[delta-low],[high-delta]],fmt="o",capsize=5,
                    color="#09675e",lw=2.4,ms=8)
    ax.set_yticks(range(len(bands)),[b[0] for b in bands])
    ax.set_ylim(len(bands)-.5,-.5)
    ticks=[-.005,0,.005,.010,.015,.020,.025]
    ax.set_xticks(ticks,[f"{x:.3f}".replace(".",",") for x in ticks])
    ax.set_xlabel("Diferencia media de Dice · primera red menos segunda")
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(loc="lower center",bbox_to_anchor=(.5,1),ncol=2,frameon=False)
    fig.savefig(assets/"intervalos_pareados.png",dpi=160)
    plt.close(fig)
    adjusted = {}
    maximum = 0
    for rank, (p, row, _) in enumerate(sorted(comparisons, key=lambda x:x[0])):
        maximum = max(maximum, min(1, (3-rank)*p))
        adjusted[row[0]] = maximum
    sections += ["## Diferencias pareadas",
        "![Diferencias medias con sus intervalos pareados del 95 % por grupos y el margen práctico positivo](figuras/intervalos_pareados.png)",
        "El punto es la diferencia media; la barra es el IC del 95 % al remuestrear los mismos grupos con las cinco semillas fijas. A la derecha de cero, la media favorece a la primera red. Para FCN frente a U-Net, la barra completa queda antes de la línea de +0,02. Para U-Net frente a SegNet, la barra atraviesa cero.",
        table(["Primer modelo menos segundo","Δ Dice","IC 95 % grupos",
        "d por grupo","IC 95 % de d","P(superior)","p Shapiro (semillas)","p exacto","p Holm","MMD aprox."],
        [row+[fmt(adjusted[row[0]]),fmt(mmd)] for _,row,mmd in comparisons])]
    ranks = stats.rankdata(means[:,:,0], axis=0)
    # Con tres modelos y cinco bloques se pueden enumerar las 6⁵ asignaciones.
    observed = np.sum((ranks.sum(axis=1)-10)**2)
    possible = np.asarray(list(product(list(permutations([1,2,3])),repeat=5)))
    exact = np.mean(np.sum((possible.sum(axis=1)-10)**2,axis=1) >= observed)
    sections.append(f"Contraste global de Friedman por semilla: Q = {fmt(observed/5)}, p exacto = {fmt(exact)} al permutar los rangos de las tres redes dentro de cada uno de los cinco bloques (7.776 asignaciones bajo intercambiabilidad).")
    conditional = []
    for i,j in combinations(range(3),2):
        d = group[i]-group[j]
        conditional.append((signed_rank_exact(d), LABEL[i]+" − "+LABEL[j],
                            (stats.norm.ppf(1-.05/6)+stats.norm.ppf(.8))*d.std(ddof=1)/np.sqrt(len(d))))
    maximum = 0
    group_rows = []
    for rank,(p,name,mmd) in enumerate(sorted(conditional)):
        maximum = max(maximum,min(1,(3-rank)*p))
        group_rows.append([name,fmt(p),fmt(maximum),fmt(mmd)])
    sections += ["### Contrastes sobre grupos con semillas fijas",
                 "Para la pregunta de generalización a otros grupos, se contrasta el centro de las diferencias de Dice de los 122 grupos, promediadas sobre las mismas cinco semillas. Wilcoxon bilateral supone independencia entre grupos y simetría de sus diferencias bajo la hipótesis nula. Se obtiene la distribución exacta por asignación de signos a los rangos absolutos, omitiendo ceros y promediando empates. Holm se aplica a esta familia de tres contrastes. La MMD para grupos usa la misma aproximación normal de planificación con la DE de las diferencias por grupo y 122 bloques. Un IC para una media y un contraste de rangos pueden dar lecturas diferentes si las diferencias son asimétricas.",
                 table(["Contraste","p exacto (grupos)","p Holm (grupos)","MMD aprox. (grupos)"], group_rows)]
    sections += ["![Cuantiles de las cinco diferencias pareadas frente a cuantiles normales](figuras/qq_semillas.png)",
                 "Los extremos de cada gráfico dependen de apenas una semilla. Con cinco puntos, Shapiro–Wilk y el gráfico Q–Q tienen poca capacidad diagnóstica. Se eligió la prueba bilateral exacta de rangos con signo, que supone simetría de las diferencias entre semillas y evita estimar una varianza común entre redes."]
    prefix = "IMO_colab_resultados_v1/continuacion_cpu_gpu/"
    run_rows = []
    with tarfile.open(ROOT/"piloto_T4_final.tar.gz") as tar:
        training = json.load(tar.extractfile(prefix+"resumen.json"))
        for i, m in enumerate(MODELS):
            for k, seed in enumerate(SEEDS):
                rec = next(r for r in training if (r["model"],r["seed"]) == (m,seed))
                folder = f"{m}_seed{seed}_lr0.001"
                local = ROOT/"resultados/piloto_local_v1"/folder/"history.json"
                cpu = json.loads(local.read_text()) if local.exists() else []
                hist = cpu if len(cpu)==rec["epochs_completed"] else json.load(tar.extractfile(prefix+folder+"/history.json"))
                assert len(hist) == rec["epochs_completed"] and hist[:len(cpu)] == cpu and rec["completed"]
                cpu_min = sum(h["seconds"] for h in cpu)/60
                gpu_min = sum(h["seconds"] for h in hist[len(cpu):])/60
                best = max(hist, key=lambda h:h["val_group_Dice"])
                run_rows.append([LABEL[i], seed,rec["parameters"],len(hist),best["epoch"],
                                 fmt(best["val_group_Dice"]),fmt(means[i,k,0]),f"{cpu_min:.1f}",f"{gpu_min:.1f}"])
    sections += ["## Presupuesto observado por corrida", table(["Red","Semilla","Parámetros","Épocas","Mejor época",
                            "Dice val.","Dice test","CPU (min)","T4 (min)"], run_rows),
                 "Se sumaron los segundos registrados por época completa. La migración de SegNet con semilla 23 conserva la parte local del historial. Los totales excluyen interrupciones y transferencias. Las diferencias de duración mezclan arquitectura, hardware y parada temprana.",
                 "Dice medio de los mejores checkpoints de validación por red: " +
                 ", ".join(LABEL[i]+" "+fmt(np.mean([r["best_val_group_Dice"] for r in training if r["model"]==m])) for i,m in enumerate(MODELS))+".",
                 "Huella SHA-256 del manifiesto de test: `"+meta["frozen_files"]["test.csv"]+"`."]
    examples = sorted((float(r["Dice"]),r["id"]) for r in detail if r["model"]=="fcn" and r["seed"]=="11")
    sections += ["## Aciertos y errores", "Se fijó FCN con semilla 11 y se eligieron tres posiciones en su distribución de Dice por imagen. Son ejemplos descriptivos escogidos después del test. En un empate se usa el identificador como desempate.",
                 table(["Posición","Imagen","Dice de FCN"], [[label,case[1],fmt(case[0])] for label,case in zip(["Mínimo","Central (posición 110)","Máximo"],[examples[0],examples[len(examples)//2],examples[-1]])]),
                 "La selección depende de FCN. Las mismas imágenes permiten comparar las otras dos redes, pero no representan necesariamente sus propios cuantiles. Un caso central por rango puede tener Dice alto si la distribución es asimétrica."]
    (BOOK/"tablas_resultados.inc").write_text("\n\n".join(sections)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
