"""Entrena UNetAT (el modelo original de E3) con el arnes del benchmark E2.

Reutiliza, importandolos por ruta desde el repositorio de E2 (solo lectura), el cargador de datos y los
utilidades del arnes: dataset_ready (partition, load_example, metrics) y experimento_e2_colab (setup_seed,
training_batches, make_optimizer, rng_state, restore_rng, atomic_json, atomic_torch, sha, digest, environment,
verify_sources). Lo que NO se puede reutilizar tal cual y se reescribe aqui, igual en lo esencial:
- train_model: construye el modelo desde MODELS_E2 y calcula la perdida con compute_loss, que no conocen a UNetAT.
  Aqui es el mismo bucle (misma semilla y orden de lotes por epoca, mismo optimizador, mismas epocas y paciencia,
  mismos checkpoints) con perdida_db sobre las etiquetas de DBNet o de DBNet-AT.
- evaluate_validation: calculaba el Dice con P >= 0,5. Aqui la mascara sale del posproceso de la variante
  (para dbnet_at, C2) y se promedia por grupo como alla; mismo fork_rng con la semilla 20261118.
- verify_data y el preflight de E2: leen las imagenes desde repo_dl_imo/e2/VOC2012, que no existe en el
  repositorio de solo lectura; aqui se verifican los mismos SHA-256 de imagenes y mascaras contra los CSV, leyendo
  las imagenes desde --voc. El preflight de E2 (preflight.json) solo cubre sus 10 modelos; aqui hay uno propio:
  un forward y backward con perdida finita antes de entrenar, anotado en preflight_original.json.

Receta de la U-Net compacta de E2 (config/experimento_e2.json, modelos.unet): Adam, lr 1e-3, weight_decay 0, lote 2,
30 epocas, paciencia 5, entrada 256, sin preentrenamiento, sin aumentos (load_example no los hace), seleccion por el
Dice medio por grupo en validacion; aqui, despues del posproceso.

Uso en GPU (Colab):
    python entrenar_original.py --variante dbnet dbnet_at --seeds 11 23 37 53 71 --output /content/drive/MyDrive/e3_original
Humo en CPU:
    python entrenar_original.py --cpu-smoke --allow-local-output --max-images 8 --epochs 2 --seeds 11 \
        --variante dbnet_at --output C:\\Proyectos\\estudio\\datos\\deep_learning\\e3_humo
Las otras 6 variantes de la ablacion 2^3 no se entrenan: salen por inferencia con los mismos checkpoints
(geometria_at.VARIANTES).
"""
import argparse
import copy
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

E3 = Path(__file__).resolve().parents[1]
E2_SCRIPTS = Path(r"C:\Proyectos\estudio\datos\deep_learning\repo_dl_imo\e2\tesis_imo_para_codex\scripts")
VOC = Path(r"C:\Proyectos\estudio\datos\hazmat13_drive")
SEMILLA_VALIDACION = 20261118  # la de evaluate_validation de E2
CLAVES_BINARIAS = ("db_gt", "db_mask", "db_umbral_mask")  # 0/1: se guardan en uint8
H = DS = G = UNetAT = perdida_db = None  # se importan en cargar_arnes()


def cargar_arnes(e2_scripts):
    """Importa el arnes de E2 por ruta y los modulos de E3. Falla con un mensaje claro si falta algo."""
    global H, DS, G, UNetAT, perdida_db
    e2_scripts = Path(e2_scripts).resolve()
    if not (e2_scripts / "experimento_e2_colab.py").exists():
        raise FileNotFoundError(f"No esta el arnes de E2 en {e2_scripts}")
    for ruta in (str(e2_scripts), str(E3)):
        if ruta not in sys.path:
            sys.path.insert(0, ruta)
    import experimento_e2_colab as h
    import dataset_ready as ds
    from src import geometria_at as g
    from src.original_model import UNetAT as red, perdida_db as perdida
    H, DS, G, UNetAT, perdida_db = h, ds, g, red, perdida
    return e2_scripts


# ------------------------------------------------------------------------------------------ datos y trazabilidad
def firma_codigo(e2_scripts):
    """SHA-256 conjunto del codigo de E3 (src y este script) y de los archivos del arnes de E2 que se usan."""
    e3 = {str(p.relative_to(E3)): H.sha(p) for p in sorted((E3 / "src").glob("*.py"))}
    e3["training/entrenar_original.py"] = H.sha(Path(__file__))
    e2 = {n: H.sha(Path(e2_scripts) / n) for n in
          ("experimento_e2_colab.py", "dataset_ready.py", "modelos_base.py", "preparar_base.py", "inicio_voc.py")}
    return {"e3": H.digest(e3), "e2_arnes": H.digest(e2), "opencv": G.cv2.__version__}


def verificar_datos(filas, voc):
    """Lo que hace verify_data de E2, con las imagenes leidas desde voc: sin fuga de grupos entre particiones
    (del test solo se leen los metadatos del CSV, como E2), SHA-256 de imagen y mascara de cada fila usada y
    normalizacion a 256. Devuelve el resumen de hashes de los CSV y de las filas usadas."""
    base = H.PROJECT / "resultados/preentrenamiento"
    firma = {f: H.sha(base / f) for f in ["train.csv", "val.csv", "test.csv", "normalization_train.json"]}
    grupos = {s: {r["group_id"] for r in DS.partition(s, allow_test=(s == "test"))} for s in ("train", "val", "test")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if grupos[a] & grupos[b]:
            raise RuntimeError(f"Fuga de grupos entre {a}/{b}")
    usados = {}
    for particion, rows in filas.items():
        if not rows:
            raise RuntimeError(f"Particion vacia: {particion}")
        for row in rows:
            for ruta, esperado in ((Path(voc) / row["image_path"], row["image_sha256"]),
                                   (H.PROJECT / row["binary_mask_path"], row["binary_mask_sha256"])):
                if H.sha(ruta) != esperado:
                    raise RuntimeError(f"Datos modificados: {ruta}")
            usados[row["id"]] = row["image_sha256"]
    norma = json.loads((base / "normalization_train.json").read_text())
    if norma["size"] != 256:
        raise RuntimeError("Normalizacion no corresponde a 256")
    return H.digest({"csv": firma, "filas_usadas": usados})


class Datos:
    """Imagenes de una particion cargadas en RAM (load_example no hace aumentos: es determinista) y, para train,
    los objetivos de cada variante, calculados una vez por imagen."""

    def __init__(self, filas, voc, objetivos=None, nombre=""):
        self.filas, self.x, self.y, self.v, self.obj = filas, {}, {}, {}, {}
        t0 = time.perf_counter()
        for fila in filas:
            x, y, v = DS.load_example(fila, data_root=voc)
            self.x[fila["id"]], self.y[fila["id"]], self.v[fila["id"]] = x, y, v
            if objetivos is not None:
                o = objetivos(y, v)
                self.obj[fila["id"]] = {k: (a.astype(np.uint8) if k in CLAVES_BINARIAS else a) for k, a in o.items()}
        print(f"[datos] {nombre}: {len(filas)} imagenes en RAM ({time.perf_counter() - t0:.1f} s)", flush=True)

    def lote(self, filas, device):
        """(x, objetivos) como tensores. x: (B, 3, H, W); cada objetivo: (B, 1, H, W) float32."""
        ids = [f["id"] for f in filas]
        x = torch.from_numpy(np.stack([self.x[i] for i in ids])).to(device)
        obj = {k: torch.from_numpy(np.stack([self.obj[i][k] for i in ids]).astype(np.float32)).to(device)
               for k in self.obj[ids[0]]}
        return x, obj


# ------------------------------------------------------------------------------------------ validacion
@torch.no_grad()
def evaluar_validacion(modelo, datos, variante, device):
    """Como evaluate_validation de E2 (Dice por imagen, promedio por grupo, luego promedio de los grupos), pero la
    mascara sale de predict_mask (posproceso de la variante) en lugar de P >= 0,5."""
    modelo.eval()
    agrupado, detalle = defaultdict(list), []
    dispositivos = [torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=dispositivos):
        torch.manual_seed(SEMILLA_VALIDACION)
        for fila in datos.filas:
            i = fila["id"]
            x = torch.from_numpy(datos.x[i][None]).to(device)
            mascara = modelo.predict_mask(x, variante, valida=datos.v[i][None])[0]
            puntajes = DS.metrics(mascara.astype(np.float32), datos.y[i], datos.v[i])
            agrupado[fila["group_id"]].append(puntajes["Dice"])
            detalle.append({"id": i, "group_id": fila["group_id"], **puntajes})
    return float(np.mean([np.mean(v) for v in agrupado.values()])), detalle


# ------------------------------------------------------------------------------------------ entrenamiento
def preflight(variante, config, identity, train, device, destino):
    """Un forward y backward con perdida y gradientes finitos antes de entrenar (sustituye al preflight de E2)."""
    H.setup_seed(0)
    modelo = UNetAT().to(device).train()
    x, obj = train.lote(train.filas[:2], device)
    perdida = perdida_db(modelo(x), obj)
    perdida.backward()
    ok = bool(torch.isfinite(perdida)) and all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                                                for p in modelo.parameters())
    informe = {"variante": variante, "status": "passed" if ok else "failed", "perdida": float(perdida.detach()),
               "identity": identity, "config_hash": H.digest(config), "environment": H.environment()}
    H.atomic_json(Path(destino) / f"preflight_original_{variante}.json", informe)
    if not ok:
        raise RuntimeError(f"Preflight fallido ({variante}): perdida {float(perdida.detach())}")


def entrenar(variante, semilla, output, config, identity, device, train, val, parada=None):
    """train_model de E2 con UNetAT. Mismo formato de config.json, last.pt, best_epoch_XXX.pt, history.json y
    result.json (fingerprint, best_file, last_sha256, best_sha256)."""
    nombre = f"unetat_{variante}"
    H.setup_seed(semilla)
    dest = Path(output) / f"{nombre}_seed{semilla}"
    dest.mkdir(parents=True, exist_ok=True)
    spec = {"name": nombre, "seed": semilla, "config": config, "identity": identity, "environment": H.environment(),
            "modelo": {"k": 50.0, "cabeza": "1x1", "g0": G.G0 if variante == "dbnet_at" else 0.0,
                       "perdida": "5*Ls + 10*Lt + Lb"}}
    huella = H.digest(spec)
    cfg_file = dest / "config.json"
    if cfg_file.exists():
        if json.loads(cfg_file.read_text())["fingerprint"] != huella:
            raise RuntimeError(f"Corrida incompatible: {dest}. Usa una carpeta de experimento nueva.")
    else:
        H.atomic_json(cfg_file, {"fingerprint": huella, **spec})
    last = dest / "last.pt"
    if (dest / "result.json").exists():
        res = json.loads((dest / "result.json").read_text())
        if res.get("fingerprint") != huella or not res.get("completed"):
            raise RuntimeError("Resultado incompatible")
        if not last.exists() or not (dest / res["best_file"]).exists():
            raise RuntimeError("Faltan checkpoints de corrida completada")
        if H.sha(last) != res["last_sha256"] or H.sha(dest / res["best_file"]) != res["best_sha256"]:
            raise RuntimeError("Checkpoint completado corrupto/modificado")
        return res
    modelo = UNetAT().to(device)  # tras setup_seed: mismos pesos iniciales de codificador/decodificador que la U-Net de E2
    optimizador = H.make_optimizer(modelo, config["models"]["unet"])
    inicio, mejor, sin_mejora, historia, mejor_archivo = 0, -1.0, 0, [], None
    if last.exists():
        guardado = torch.load(last, map_location="cpu", weights_only=False)
        if guardado["fingerprint"] != huella:
            raise RuntimeError("Checkpoint incompatible")
        modelo.load_state_dict(guardado["model"])
        optimizador.load_state_dict(guardado["optimizer"])
        inicio, mejor, sin_mejora, historia, mejor_archivo = (guardado["epoch"] + 1, guardado["best"], guardado["stale"],
                                                              guardado["history"], guardado["best_file"])
        if not (dest / mejor_archivo).exists():
            raise RuntimeError("Falta mejor checkpoint referenciado")
        H.restore_rng(guardado["rng"])
    if parada is not None and inicio >= parada and inicio < config["epochs"] and sin_mejora < config["patience"]:
        return {"completed": False, "paused": True, "epochs_completed": inicio}
    for epoca in range(inicio, config["epochs"]):
        if sin_mejora >= config["patience"]:
            break
        modelo.train()
        t0, suma, vistas = time.perf_counter(), 0.0, 0
        for filas in H.training_batches(train.filas, semilla, epoca, config["batch_size"]):
            x, obj = train.lote(filas, device)
            optimizador.zero_grad(set_to_none=True)
            perdida = perdida_db(modelo(x), obj)
            if not torch.isfinite(perdida):
                raise RuntimeError("Perdida no finita")
            perdida.backward()
            optimizador.step()
            suma += perdida.item() * len(filas)
            vistas += len(filas)
        t_entreno = time.perf_counter() - t0
        puntaje, detalle = evaluar_validacion(modelo, val, variante, device)
        if not np.isfinite(puntaje):
            raise RuntimeError("Validacion no finita")
        if puntaje > mejor:
            mejor, sin_mejora = puntaje, 0
            mejor_archivo = f"best_epoch_{epoca + 1:03d}.pt"
            H.atomic_torch(dest / mejor_archivo, {"model": modelo.state_dict(), "epoch": epoca, "fingerprint": huella,
                                                  "validation": detalle})
        else:
            sin_mejora += 1
        historia.append({"epoch": epoca + 1, "train_loss": suma / vistas, "val_group_Dice": puntaje,
                         "seconds": time.perf_counter() - t0, "seconds_train": t_entreno})
        # last apunta a un best inmutable ya escrito: un corte no cambia su significado
        H.atomic_torch(last, {"fingerprint": huella, "epoch": epoca, "model": modelo.state_dict(),
                              "optimizer": optimizador.state_dict(), "best": mejor, "stale": sin_mejora,
                              "history": historia, "best_file": mejor_archivo, "rng": H.rng_state()})
        H.atomic_json(dest / "history.json", historia)
        for viejo in dest.glob("best_epoch_*.pt"):
            if viejo.name != mejor_archivo:
                viejo.unlink()
        print(f"{nombre} seed={semilla} epoch={epoca + 1} loss={suma / vistas:.4f} val={puntaje:.5f} "
              f"({historia[-1]['seconds']:.1f} s; entrenamiento {t_entreno:.1f} s)", flush=True)
        if (parada is not None and epoca + 1 >= parada
                and epoca + 1 < config["epochs"] and sin_mejora < config["patience"]):
            return {"completed": False, "paused": True, "epochs_completed": epoca + 1}
    res = {"completed": True, "fingerprint": huella, "model": nombre, "variante": variante, "seed": semilla,
           "best_val_group_Dice": mejor, "epochs_completed": len(historia), "best_file": mejor_archivo,
           "parameters": sum(p.numel() for p in modelo.parameters()),
           "last_sha256": H.sha(last), "best_sha256": H.sha(dest / mejor_archivo)}
    H.atomic_json(dest / "result.json", res)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--variante", nargs="+", default=["dbnet", "dbnet_at"], choices=["dbnet", "dbnet_at"],
                    help="brazos que se entrenan; las otras 6 variantes salen por inferencia")
    ap.add_argument("--seeds", nargs="+", type=int, help="por defecto, las del config de E2")
    ap.add_argument("--output", required=True)
    ap.add_argument("--config", default=None, help="por defecto, config/experimento_e2.json del arnes de E2")
    ap.add_argument("--allow-local-output", action="store_true")
    ap.add_argument("--cpu-smoke", action="store_true", help="humo en CPU; exige --max-images")
    ap.add_argument("--max-images", type=int, help="usa solo las primeras N imagenes de train y de val (humo)")
    ap.add_argument("--epochs", type=int, help="sobrescribe las epocas del config (humo)")
    ap.add_argument("--pause-after-epoch", type=int,
                    help="pausa después de esta época absoluta sin cambiar la receta; omitir para reanudar")
    ap.add_argument("--voc", default=str(VOC), help="carpeta que contiene VOC2012/ (las imagenes)")
    ap.add_argument("--e2-scripts", default=str(E2_SCRIPTS))
    args = ap.parse_args(argv)
    if args.pause_after_epoch is not None and args.pause_after_epoch < 1:
        ap.error("--pause-after-epoch debe ser positivo")

    e2_scripts = cargar_arnes(args.e2_scripts)
    output = Path(args.output).resolve()
    if not args.allow_local_output:
        import subprocess
        subprocess.run(["mountpoint", "-q", "/content/drive"], check=True)
        if not output.is_relative_to(Path("/content/drive/MyDrive")):
            raise RuntimeError("Salida debe estar en Drive")
    if args.cpu_smoke:
        if not args.max_images or args.max_images > 64:
            raise RuntimeError("--cpu-smoke exige --max-images (hasta 64): una corrida completa en CPU no es un humo")
        device = torch.device("cpu")
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("Este entrenamiento requiere GPU; para un humo en CPU usa --cpu-smoke --max-images N")
        device = torch.device("cuda")
        if args.max_images or args.epochs:
            raise RuntimeError("--max-images y --epochs son solo para --cpu-smoke")
    config = json.loads(Path(args.config or H.PROJECT / "config/experimento_e2.json").read_text())
    if (config["pretrained"] or config["size"] != 256 or config["batch_size"] < 2 or config["threshold"] != .5
            or config["epochs"] < 1 or config["patience"] < 1
            or config["loss"] != "0.5_BCEWithLogits_plus_0.5_softDice_per_image_valid"):
        raise ValueError("Configuracion no soportada")
    config = copy.deepcopy(config)
    if args.epochs:
        config["epochs"] = args.epochs
    if args.max_images:
        config["max_images"] = args.max_images  # entra en la huella: un humo nunca se confunde con una corrida completa

    H.verify_sources()
    train_filas, val_filas = DS.partition("train"), DS.partition("val")
    if args.max_images:
        train_filas, val_filas = train_filas[:args.max_images], val_filas[:args.max_images]
    identity = {**firma_codigo(e2_scripts),
                "data": verificar_datos({"train": train_filas, "val": val_filas}, args.voc)}
    output.mkdir(parents=True, exist_ok=True)
    val = Datos(val_filas, args.voc, nombre="val")
    resultados = []
    for variante in args.variante:
        train = Datos(train_filas, args.voc, G.objetivos_de(variante), nombre=f"train ({variante}, objetivos C1)"
                      if variante == "dbnet_at" else f"train ({variante}, objetivos DBNet)")
        preflight(variante, config, identity, train, device, output)
        for semilla in args.seeds or config["seeds"]:
            resultados.append(entrenar(variante, semilla, output, config, identity, device, train, val,
                                       parada=args.pause_after_epoch))
    print(json.dumps(resultados, indent=1))
    return resultados


if __name__ == "__main__":
    main()
