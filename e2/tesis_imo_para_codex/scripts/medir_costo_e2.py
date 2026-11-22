"""Costo computacional de las 10 arquitecturas: parametros, FLOPs, memoria pico y latencia por imagen.

  python scripts/medir_costo_e2.py medir [--modelos a,b] [--hilos 1,6] [--reps 30] [--calentamiento 5] [--salida DIR]
  python scripts/medir_costo_e2.py tabla                 # une los JSON de la carpeta en costo_e2.csv/.md

Sin pesos entrenados: se instancia cada red con inicializacion aleatoria (el costo no depende de los pesos).
Entrada del benchmark: 1 imagen RGB de 256x256 (config/experimento_propuesto.json, local_profile.image_size).
- FLOPs: torch.utils.flop_counter.FlopCounterMode (cuenta convoluciones, matmul, bmm y atencion; 1 MAC = 2 FLOPs;
  no cuenta normalizaciones, activaciones ni interpolaciones). Es un conteo de operaciones, no un tiempo.
- Memoria pico: cada modelo corre en un subproceso nuevo. Se reporta el pico del proceso (PeakWorkingSetSize) y el
  incremento del conjunto de trabajo durante la primera inferencia, muestreado cada 1 ms, respecto al modelo ya cargado.
- Latencia: batch 1, torch.no_grad, calentamiento, N repeticiones; mediana e IQR. Hilos fijados con
  torch.set_num_threads (se mide cada valor de --hilos). Hardware registrado en cada JSON.
Cada modelo escribe un JSON propio; si falla (VM-UNet sin CUDA, TransUNet sin ml_collections) escribe el motivo.
En GPU (Colab) el mismo script mide con --dispositivo cuda (latencia con sincronizacion, memoria con
torch.cuda.max_memory_allocated).
"""
import argparse
import ctypes
import json
import platform
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
MODELOS = ["fcn", "segnet", "unet", "deeplabv3plus", "unetplusplus", "pidnet_s", "segnext_t", "segformer_b0",
           "transunet_2d", "vmunet"]
TAM = 256


class _PMC(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


def pico_mib():
    """Pico del conjunto de trabajo del proceso (MiB). Solo Windows."""
    if not hasattr(ctypes, "windll"):
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    k32, ps = ctypes.windll.kernel32, ctypes.windll.psapi
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    ps.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
    pmc = _PMC()
    pmc.cb = ctypes.sizeof(pmc)
    ps.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
    return pmc.PeakWorkingSetSize / 2 ** 20


def actual_mib():
    """Conjunto de trabajo actual del proceso (MiB)."""
    if not hasattr(ctypes, "windll"):
        return pico_mib()
    k32, ps = ctypes.windll.kernel32, ctypes.windll.psapi
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    ps.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
    pmc = _PMC()
    pmc.cb = ctypes.sizeof(pmc)
    ps.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
    return pmc.WorkingSetSize / 2 ** 20


def memoria_inferencia_fria(modelo, x, fn):
    """Incremento del conjunto de trabajo durante la PRIMERA inferencia (muestreo cada 1 ms) sobre el valor previo.
    Se mide antes de cualquier otra pasada, porque el conjunto de trabajo no se encoge al reutilizar memoria."""
    import threading
    base = actual_mib()
    mx, parar = [base], threading.Event()

    def muestrear():
        while not parar.is_set():
            mx[0] = max(mx[0], actual_mib())
            time.sleep(0.001)
    t = threading.Thread(target=muestrear, daemon=True)
    t.start()
    fn(modelo(x))
    parar.set()
    t.join()
    return base, mx[0]


def hardware():
    import torch
    info = {"so": platform.platform(), "python": platform.python_version(), "torch": torch.__version__,
            "hilos_intraop_por_defecto": torch.get_num_threads(), "gpu": torch.cuda.is_available()}
    if torch.cuda.is_available():
        info["gpu_nombre"] = torch.cuda.get_device_name(0)
    elif platform.system() == "Windows":
        try:
            out = subprocess.run(["powershell", "-NoProfile", "-c",
                                  "(Get-CimInstance Win32_Processor | Select -First 1 Name,NumberOfCores,"
                                  "NumberOfLogicalProcessors | ConvertTo-Json -Compress)"],
                                 capture_output=True, text=True, timeout=30).stdout.strip()
            info["cpu"] = json.loads(out)
        except Exception:
            info["cpu"] = platform.processor()
    return info


def trabajador(nombre, hilos, reps, calent, salida, disp):
    import numpy as np
    import torch
    torch.manual_seed(0)
    torch.set_num_threads(hilos)
    cuda = disp == "cuda"
    res = {"modelo": nombre, "dispositivo": disp, "entrada": [1, 3, TAM, TAM], "hilos": hilos, "reps": reps,
           "calentamiento": calent, "hardware": hardware()}
    try:
        from modelos_e2 import MODELS_E2, main_logits
        modelo = MODELS_E2[nombre]().eval()
        if cuda:
            modelo = modelo.cuda()
    except Exception as e:  # modelo que no corre aqui: se registra el motivo
        res["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        Path(salida).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        return
    res["parametros"] = sum(p.numel() for p in modelo.parameters())
    res["parametros_entrenables"] = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    x = torch.randn(1, 3, TAM, TAM, device=disp)
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    with torch.no_grad():
        if not cuda:
            base_fria, pico_frio = memoria_inferencia_fria(modelo, x, main_logits)
            res["mem_base_modelo_cargado_mib"] = base_fria
            res["mem_pico_primera_inferencia_mib"] = pico_frio
            res["mem_inferencia_incremento_mib"] = pico_frio - base_fria
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as fc:
            main_logits(modelo(x))
        res["flops"] = int(fc.get_total_flops())
        res["flops_por_operador"] = {str(k): int(v) for k, v in fc.get_flop_counts()["Global"].items()}
        for _ in range(calent):
            main_logits(modelo(x))
        sync()
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        tiempos = []
        for _ in range(reps):
            t = time.perf_counter()
            main_logits(modelo(x))
            sync()
            tiempos.append((time.perf_counter() - t) * 1000)
    q = np.percentile(tiempos, [25, 50, 75])
    res.update({"latencia_ms_mediana": float(q[1]), "latencia_ms_q1": float(q[0]), "latencia_ms_q3": float(q[2]),
                "latencia_ms_iqr": float(q[2] - q[0]), "latencia_ms_min": float(min(tiempos)),
                "latencia_ms_max": float(max(tiempos)), "latencias_ms": [round(t, 3) for t in tiempos],
                "pico_proceso_mib": pico_mib(),
                "mem_nota": "pico_proceso = PeakWorkingSetSize de todo el proceso (incluye torch y la construccion del "
                            "modelo). mem_inferencia_incremento = alza del conjunto de trabajo en la primera inferencia, "
                            "muestreado cada 1 ms (cota inferior: un pico mas corto que 1 ms puede escaparse)"})
    if cuda:
        res["gpu_mem_pico_mib"] = torch.cuda.max_memory_allocated() / 2 ** 20
    Path(salida).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")


def medir(a):
    dest = Path(a.salida)
    dest.mkdir(parents=True, exist_ok=True)
    for nombre in a.modelos.split(","):
        for h in [int(v) for v in a.hilos.split(",")]:
            out = dest / f"{nombre}_{a.dispositivo}_h{h}.json"
            print(f"{nombre} {a.dispositivo} hilos={h} ...", flush=True)
            subprocess.run([sys.executable, "-W", "ignore", __file__, "_w", nombre, str(h), str(a.reps),
                            str(a.calentamiento), str(out), a.dispositivo], check=False)
            if out.exists():
                r = json.loads(out.read_text(encoding="utf-8"))
                print("  ", r.get("error") or f"{r['parametros']:,} par, {r['flops']/1e9:.2f} GFLOPs, "
                      f"{r['latencia_ms_mediana']:.1f} ms")
            else:
                print("   sin salida (el subproceso murio)")


def tabla(a):
    dest = Path(a.salida)
    filas = []
    for nombre in MODELOS:
        for p in sorted(dest.glob(f"{nombre}_*_h*.json")):
            filas.append(json.loads(p.read_text(encoding="utf-8")))
    cab = "modelo,dispositivo,hilos,parametros,gflops,latencia_ms_mediana,latencia_ms_iqr,pico_proceso_mib,mem_inferencia_incremento_mib,nota"
    csv = [cab]
    md = ["| Modelo | Disp. | Hilos | Parámetros | GFLOPs | Latencia mediana (ms) | IQR (ms) | Pico proceso (MiB) | Δ inferencia (MiB) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in filas:
        if "error" in r:
            csv.append(f"{r['modelo']},{r['dispositivo']},{r['hilos']},,,,,,,\"{r['error']}\"")
            md.append(f"| {r['modelo']} | {r['dispositivo']} | {r['hilos']} | no corre aquí | | | | | |")
            continue
        csv.append(f"{r['modelo']},{r['dispositivo']},{r['hilos']},{r['parametros']},{r['flops']/1e9:.4f},"
                   f"{r['latencia_ms_mediana']:.2f},{r['latencia_ms_iqr']:.2f},{r['pico_proceso_mib']:.0f},"
                   f"{r['mem_inferencia_incremento_mib']:.0f},")
        md.append(f"| {r['modelo']} | {r['dispositivo']} | {r['hilos']} | {r['parametros']:,} | {r['flops']/1e9:.2f} | "
                  f"{r['latencia_ms_mediana']:.1f} | {r['latencia_ms_iqr']:.1f} | {r['pico_proceso_mib']:.0f} | "
                  f"{r['mem_inferencia_incremento_mib']:.0f} |")
    (dest / "costo_e2.csv").write_text("\n".join(csv) + "\n", encoding="utf-8")
    (dest / "costo_e2.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "_w":
        trabajador(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6], sys.argv[7])
        sys.exit(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("modo", choices=["medir", "tabla"])
    ap.add_argument("--modelos", default=",".join(MODELOS))
    ap.add_argument("--hilos", default="1,6")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--calentamiento", type=int, default=5)
    ap.add_argument("--dispositivo", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--salida", default=str(RAIZ / "resultados" / "costo_e2"))
    a = ap.parse_args()
    (medir if a.modo == "medir" else tabla)(a)
