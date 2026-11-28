---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(sec-reproducibilidad)=
# Lista de verificación de reproducibilidad

## Entorno de esta construcción

```{code-cell} python3
:tags: [hide-input]
# Versiones del software con el que se construyó este libro y huella de la tabla principal.
import hashlib, platform, sys
from importlib import metadata
from pathlib import Path

print("Python:", sys.version.split()[0], "| sistema:", platform.platform(), "| CPU:", platform.processor() or "n/d")
for p in ["numpy", "pandas", "matplotlib", "jupyter-book", "myst-nb", "sphinx"]:
    try:
        print(f"{p}: {metadata.version(p)}")
    except metadata.PackageNotFoundError:
        print(f"{p}: no instalado")
csv = Path("..") / "data" / "tabla_maestra.csv"
print("SHA-256 tabla_maestra.csv:", hashlib.sha256(csv.read_bytes()).hexdigest())
```

## Estado operativo vigente

La secuencia v4 y el lanzador único conservan checkpoints por época y permiten reanudar. La identidad registra la versión de OpenCV 4.11 y las huellas del código; un cambio de identidad impide reutilizar silenciosamente un checkpoint. La prueba local reducida de entrenamiento y reanudación exacta aprobó en CPU; no acredita por sí sola determinismo entre GPU o plataformas.

La campaña prevista comprende 120 corridas: 20 del original y 100 de rivales. Las 55 nuevas de rivales completan las compactas y corrigen SegFormer; no basta añadir cinco semillas indiscriminadamente a cada resultado histórico. La auditoría y las predicciones de validación son etapas separadas. Los tiempos finales, aceleradores realmente asignados y resultados deben leerse de sus registros, sin asumir que todas las sesiones usarán L4.

El libro usa resultados livianos y no entrena modelos al compilar. La demostración de test contiene 2.180 máscaras verificadas de diez redes; su inventario está en `data/demo_test_verificacion.json`. Las fotografías permanecen en el paquete local. La aplicación utiliza `http://[::1]:8051/modelos` en este equipo.

## Lista recuperada del plan inicial

La tabla siguiente conserva el plan inicial como antecedente. Sus recetas uniformes, estimaciones de cómputo, hardware previsto y estados pendientes no sustituyen la secuencia v4 ni un registro de ejecución. El tag de congelamiento y la reproducción independiente todavía deben verificarse.


```{list-table} Lista de verificación de reproducibilidad
:header-rows: 1
:name: tab-reproducibilidad

* - Ítem
  - Estado
  - Dónde
* - Versiones de software
  - Entrenamiento: `requirements-train.txt` (torch 2.5.1, numpy 1.26.4, timm 1.0.15, ...). Dashboard: `requirements.txt`. Libro: `book/requirements.txt`
  - Archivos citados
* - Hardware
  - Entrenamiento en GPU NVIDIA L4 (Colab) para las semillas del E3; latencias del E2 en CPU AMD Ryzen 5 PRO 6650U y GPU (nota de `tabla_maestra_meta.json`). VM-UNet exige CUDA
  - `data/tabla_maestra_meta.json`
* - Semillas
  - 11, 23, 37, 53, 71, 89, 97, 101, 103, 107; reserva 127, 131, 137, 139, 149
  - `experiments/protocolo.md`, §3
* - Determinismo
  - Pendiente: indicadores `torch.use_deterministic_algorithms` y de cuDNN usados por el arnés, y prueba de dos corridas con igual semilla
  - Bloque A del plan
* - Rangos de hiperparámetros
  - Receta fija por red, sin búsqueda en los rivales (Adam, lr 1e-3, lote 2, hasta 30 épocas, paciencia 5); el original recibe lo mismo. Sensibilidad a $g_0\in\{2, 8\}$ px, exploratoria
  - `experiments/protocolo.md`, §2 y §10
* - Número de ejecuciones
  - 10 semillas por modelo (5 del E2 más 5 del E3) y por brazo del original
  - `experiments/protocolo.md`, §3
* - Dispersión
  - DE entre semillas e IC 95 % por grupos en la tabla principal
  - {ref}`sec-resultados`
* - Datos
  - HAZMAT-13: enlace y huellas en la declaración de datos; no se redistribuyen imágenes
  - `data/DATA_STATEMENT.md`
* - Código
  - Pendiente: URL del repositorio y etiqueta `e3-protocolo` (sin commit ni push hasta el visto bueno del equipo). SHA-256 de los módulos del original
  - `experiments/procedencia.md`
* - Tiempo de cómputo
  - Medido: 3,28 h de L4 por semilla de los rivales (E2). Estimado: unas 20 h de L4 para el E3 (semillas 6 a 10 de los rivales, dos brazos del original con 10 semillas, búsqueda)
  - `plan.md`, sección Cómputo
* - Energía estimada
  - Pendiente de cálculo final: energía $\approx$ horas de GPU $\times$ potencia máxima declarada del acelerador (L4: 72 W); es una cota superior, no una medición
  - Este capítulo
```

## Reproducción independiente

Procedimiento: una persona ajena al diseño construye la imagen, ejecuta el libro desde cero en el contenedor y
regenera la tabla principal; se reporta la discrepancia máxima entre su tabla y la publicada.

```{code-block} bash
docker build -t e3 .
docker run --rm -v "$PWD/book/_build:/app/book/_build" e3 jupyter-book build book/
```

Discrepancia máxima, por columna numérica $c$ de la tabla principal:

$$
\Delta_{\max} = \max_{j,\,c}\ \bigl\lvert x^{\text{ajena}}_{j,c} - x^{\text{publicada}}_{j,c}\bigr\rvert
$$ (eq-discrepancia)

```{admonition} Pendiente
:class: warning
- **Reproducción independiente:** no se ha hecho; la imagen de Docker no se ha construido (Docker no está verificado en
  esta máquina). Falta quién la ejecute, la fecha, el equipo y el $\Delta_{\max}$ medido con {eq}`eq-discrepancia`.
- **Reproducción de las semillas 6 a 10 y del original:** requiere GPU; las cifras del benchmark del E2 sí se
  regeneran desde `E3_E2_RESULTADOS` con `experiments/generar_tabla_maestra.py`.
```
