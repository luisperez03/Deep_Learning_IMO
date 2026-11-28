"""Arma matriz_novedad.csv y matriz_novedad.md.

Resultados, revisados y fecha salen de los archivos crudos de resultados/ (los escribe
buscar.py). Los relevantes y la observacion son juicio del revisor sobre titulo y
resumen y estan en CURADO, para que la matriz sea reproducible y el juicio, auditable.

Uso: python armar_matriz.py
"""
import csv
import json
from pathlib import Path

AQUI = Path(__file__).parent
RES = AQUI / "resultados"

NOMBRES = {"openalex": "OpenAlex", "arxiv": "arXiv", "semanticscholar": "Semantic Scholar (masiva)",
           "openreview": "OpenReview", "github": "GitHub (repositorios)", "huggingface": "Hugging Face Hub"}

# (fuente, cadena): (relevantes, amenaza a C1/C2, observacion)
CURADO = {
    ("openalex", "C1"): ("7 trabajos (10 registros)", "no",
        "Los relevantes detectan, clasifican o leen rótulos; ninguno segmenta con red. Nuevos frente a E2: Vision on the Move (2025), HMD-Net (2022), DOT-HAZMAT (2022), informe de lectores de placas en carretera. El resto es ruido de riesgo, logística y derecho."),
    ("openalex", "C2"): ("0", "no", "Solo ruido de transporte y química; ningún estudio de visión en español entre los 50."),
    ("openalex", "C3"): ("4", "no",
        "Mejoras de DBNet por backbone, atención, factor k adaptativo (ASM-DBNet 2026), SPP o convergencia (IDBNet, HTBNet). Ninguna cambia la regla de encogimiento por grosor según el resumen. Muchas aplicaciones industriales de texto (medidores, facturas, placas de equipos)."),
    ("openalex", "C4"): ("4 trabajos (6 registros)", "parcial",
        "Text kernel expansion (2024) y Deformable Kernel Expansion (3 registros) expanden el núcleo regresando offsets; PSENet; offsetting de polígonos con aumento de borde (2020, sin leer a fondo). Ninguno resuelve d como punto fijo de la regla de encogimiento."),
    ("openalex", "C5"): ("2", "no",
        "Ningún trabajo aplica DBNet o núcleos encogidos a rótulos hazmat. Aparece una tesis de 2026 sobre placas hazmat con YOLOv9-t (detección, clasificación y texto) y Similar Mask (2024, texto en tráfico)."),
    ("openalex", "C6"): ("2", "no",
        "Transferencia de DB fuera del texto: DB M-Net (2020, DB sobre una variante de U-Net, TC de esófago) y binarización adaptativa sobre Mask R-CNN (2021). Usan la binarización, no el encogimiento del núcleo. Posiciones 6-50: ruido de otras disciplinas."),
    ("openalex", "C7"): ("0", "no", "Ruido arqueológico (reducción de núcleos líticos). Un único registro de detección de texto sin resumen."),
    ("openalex", "C8"): ("3", "no",
        "Representaciones por transformada de distancia o esqueleto (TextMountain 2020, T-Skeleton 2024, skeleton-aware distance transform 2023) como mapas a regresar; ninguna atenúa el encogimiento del núcleo por radio inscrito según el resumen."),
    ("arxiv", "C1"): ("2", "no", "DeepHAZMAT (YOLO y GrabCut) y detección planar RGB-D (E2-0007). Solo 3 resultados."),
    ("arxiv", "C2"): ("0", "no", "0 resultados. El primer intento falló por error de red (HTTP 0) y el reintento respondió."),
    ("arxiv", "C3"): ("4", "no",
        "DBNet, DBNet++, GWNet (k adaptativo, no encogimiento) e IIM (DB en localización de multitudes). Ningún encogimiento atenuado."),
    ("arxiv", "C4"): ("1", "parcial", "Deformable Kernel Expansion (2023): encogimiento fijo r = 0,4 y offsets de vértices regresados (texto completo leído)."),
    ("arxiv", "C5"): ("0", "no", "0 resultados: el cruce mecanismo-problema está vacío en arXiv."),
    ("arxiv", "C6"): ("4", "no", "DBNet, DBNet++, OCR manuscrito árabe e IIM (transferencia de DB a localización de multitudes, 2020)."),
    ("arxiv", "C7"): ("0", "no", "0 resultados."),
    ("arxiv", "C8"): ("0", "no", "0 resultados."),
    ("semanticscholar", "C1"): ("14", "no",
        "Hazmat: DeepHAZMAT, Zhao 2013 (dos trabajos), Mohamed 2018, Zhang 2024, y varios de detección o clasificación (2008-2026, YOLO, color, SURF). Todos detectan, clasifican o reconocen; ninguno segmenta con red. Orden por citas, no por relevancia."),
    ("semanticscholar", "C2"): ("0", "no", "Normativa de transporte; sin visión."),
    ("semanticscholar", "C3"): ("4", "no", "DBNet, DBNet++, GWNet y detección de medidores con DB. Homónimos de DBNet (retinopatía, carriles) y ruido."),
    ("semanticscholar", "C4"): ("2", "parcial", "Text kernel expansion y Deformable Kernel Expansion; ambos con regresión."),
    ("semanticscholar", "C5"): ("0", "no", "DBNet en señales turísticas (texto sobre la señal), no el rótulo."),
    ("semanticscholar", "C6"): ("0", "no", "Todas las aplicaciones siguen siendo de texto (médico, facturas, libros antiguos)."),
    ("semanticscholar", "C7"): ("0", "no", "Un registro general de texto en escenas."),
    ("semanticscholar", "C8"): ("0", "no", "0 resultados."),
    ("openreview", "C1"): ("2", "no", "Placas hazmat con YOLOv5-Small y datos sintéticos (E2-0016) y Vision on the Move (2025). El resto, ruido; muchos registros son comentarios de revisión sin título."),
    ("openreview", "C2"): ("0", "no", "Sin relación."),
    ("openreview", "C3"): ("1", "no", "Texto con DB y segmentación semántica; el resto es binarización de documentos, de redes o de otra disciplina. El tope de 10.000 no es un conteo."),
    ("openreview", "C4"): ("4", "parcial", "Text kernel expansion, Kernel Adaptive Convolution (mapa de distancia), PSENet y EK-Net (ICASSP 2024, distancia de expansión regresada)."),
    ("openreview", "C5"): ("1", "no", "Solo el registro de DB con segmentación semántica."),
    ("openreview", "C6"): ("1", "no", "Solo el registro de DB con segmentación semántica."),
    ("openreview", "C7"): ("0", "no", "Sin relación."),
    ("openreview", "C8"): ("5", "parcial",
        "Reinforcement Shrink-Mask (TMM 2023), Kernel Adaptive Convolution, EK-Net, Text kernel expansion y Zoom Text Detector: kernel con offsets o mapas de distancia regresados, todos con encogimiento de la etiqueta fijo."),
    ("github", "C1"): ("0", "no", "0 repositorios con la cadena completa; la búsqueda de GitHub exige todas las palabras. Ver consultas cortas."),
    ("github", "C2"): ("0", "no", "0 repositorios."),
    ("github", "C3"): ("0", "no", "0 repositorios."),
    ("github", "C4"): ("0", "no", "0 repositorios."),
    ("github", "C5"): ("0", "no", "Tres reimplementaciones genéricas de DBNet; ninguna con rótulos."),
    ("github", "C6"): ("0", "no", "0 repositorios."),
    ("github", "C7"): ("0", "no", "0 repositorios."),
    ("github", "C8"): ("0", "no", "0 repositorios."),
    ("huggingface", "C1"): ("2", "no", "Dos conjuntos de datos de placas hazmat (hmgill/Vehicle-Hazmat-Placards, barissonmezee/un-hazmat); el resto son modelos de texto, tablas regulatorias y espacios sin relación."),
    ("huggingface", "C2"): ("0", "no", "0 resultados."),
    ("huggingface", "C3"): ("1", "no", "Modelos DBNet y DBNet++ de texto; uno para códigos de contenedor (mattkro/container-code-dbnetpp, texto, no rótulos)."),
    ("huggingface", "C4"): ("0", "no", "Los 82 resultados son modelos de difusión 'unclip', homónimos sin relación."),
    ("huggingface", "C5"): ("0", "no", "0 resultados."),
    ("huggingface", "C6"): ("0", "no", "0 resultados."),
    ("huggingface", "C7"): ("0", "no", "0 resultados."),
    ("huggingface", "C8"): ("0", "no", "0 resultados."),
}

# (clave de snowball, relevantes, observacion)
SNOW = [
    ("DBNet", "11", "Citas relevantes por mecanismo: ADNet, Reinforcement Shrink-Mask, DKE, Text kernel expansion, EK-Net y EK-Net++, Optimal Kernel (OKR), Text kernel calculation, Pull Pole Points (MTD), CentripetalText y FAST. Ver comparativa.md. Las referencias hacia atrás son PSENet, TextSnake, PixelLink y ATRR (Wang et al., 2019), sin regla adaptativa. El registro de OpenAlex es el de AAAI (1067 citas); otros registros de DBNet no se suman."),
    ("DBNetpp", "8", "Mismo subconjunto de DBNet (sin CentripetalText ni OKR en el filtro). DBNet++ cambia la fusión de escalas, no la etiqueta."),
    ("Jiang2021", "1", "Solo 16 citas. CM-Net (concentric mask, 2022) es la única de mecanismo; no se leyó el texto. Nadie cita la expansión en inglete con distancia predicha como método."),
    ("ASMTD", "0", "OpenAlex no registra citas del preprint; su sucesor, Reinforcement Shrink-Mask (TMM 2023), aparece en las citas de DBNet. Las referencias incluyen TextSnake, FCENet, ContourNet, TextRay y ABCNet."),
    ("DeepHAZMAT", "1", "Registro SSRN/OSF de 5 citas; la única de dominio cercano es Chemical Safety Sign Detection: A Survey and Benchmark (IJCNN 2022, detección)."),
    ("SegFormer", "0", "11671 citas; 0 con palabras del problema (hazmat, placard, dangerous goods, warning sign) en título o resumen. Las 75 referencias son de segmentación semántica general."),
]
SNOW_HAZMAT = {
    "DeepHAZMAT (MLWA 2021)": ("2", "HMD-Net y un sistema de identificación de sustancias peligrosas desde aeronaves no tripuladas (2022, no leído). Ningún citante segmenta rótulos con red."),
    "Zhang et al. 2024, fusion visual y textual (TAI)": ("1", "Detection of Multi-Class Hazard Signs using YOLO (2025): detección."),
    "Mohamed et al. 2018 (SSRR)": ("1", "Ídem; más el trabajo RGB-D de geometría previa (E2-0007)."),
    "Zhao et al. 2013 (GlobalSIP)": ("2", "Zhang 2024 y el trabajo RGB-D de geometría previa; clásicos o de detección."),
    "HMD-Net 2022 (Entropy)": ("2", "Dangerous Goods Detection and Warning (2025) y reidentificación de vehículos hazmat en túneles (2023); detección."),
}

FILAS_EXTRA = [
    ("2026-11-23", "Google Scholar", "C1-C8", "no accesible", "0", "0", "n/a", "HTTP 403 desde el cliente automático (2026-11-23). Queda para quien tenga navegador."),
    ("2026-11-23", "Papers with Code", "C1-C8", "no accesible", "0", "0", "n/a", "El sitio redirige a huggingface.co/papers/trending (2026-11-23). Se sustituyó por Hugging Face Papers (filas siguientes)."),
    ("2026-11-23", "Scopus", "C1-C8", "pendiente", "", "", "", "Requiere el proxy de la biblioteca; cadenas listas en pendiente_bases_suscripcion.md."),
    ("2026-11-23", "Web of Science", "C1-C8", "pendiente", "", "", "", "Ídem."),
    ("2026-11-23", "IEEE Xplore", "C1-C8", "pendiente", "", "", "", "Ídem. Aquí están los textos completos que no se pudieron leer (OKR, ADNet con fórmula, Similar Mask)."),
]


def main():
    filas = []
    for fuente in NOMBRES:
        for cid in [f"C{i}" for i in range(1, 9)]:
            f = RES / f"{fuente}__{cid}.json"
            d = json.loads(f.read_text(encoding="utf-8"))
            tot = d["total_declarado"]
            if d["estado"] != "ok":
                tot = d["estado"]
            elif fuente == "openreview" and tot == 10000:
                tot = "10000 (tope de la API)"
            rel, amen, obs = CURADO[(fuente, cid)]
            filas.append([d["fecha"][:10], NOMBRES[fuente], cid, tot, d["devueltos"], rel, amen, obs])
    # consultas cortas de GitHub y Hugging Face Papers
    g = json.loads((RES / "github__cortas.json").read_text(encoding="utf-8"))
    rel_gh = {"hazmat segmentation": "1", "hazmat placard": "3 (de 16)", "hazmat detection": "6 (de 25)",
              "dangerous goods label detection": "1 (conjunto de datos)", "hazmat-13": "1",
              "DeepHAZMAT": "1", "differentiable binarization segmentation": "0",
              "text kernel expansion": "0", "adaptive shrink ratio text detection": "0",
              "unclip_ratio adaptive": "0", "shrink ratio DBNet": "0", "IMDG label detection": "0"}
    for c in g["consultas"]:
        filas.append([g["fecha"][:10], "GitHub (consulta corta)", "cortas", c["total_declarado"], c["devueltos"],
                      rel_gh[c["consulta"]], "no", f"Consulta: {c['consulta']}"])
    h = json.loads((RES / "huggingface_papers.json").read_text(encoding="utf-8"))
    rel_hf = {"hazmat": ("0", "Sin artículos."),
              "differentiable binarization": ("2", "DBNet y DBNet++ entre los primeros; el resto, binarización de redes, de documentos o de rasterizadores."),
              "text kernel expansion shrink": ("2", "PSENet y similares; ruido de modelos de lenguaje."),
              "hazardous material placard segmentation": ("0", "Segmentación de residuos y humo; ningún rótulo."),
              "adaptive shrink ratio text detection": ("2", "PSENet y FAST; sin regla de encogimiento adaptativa.")}
    for c in h["consultas"]:
        r, o = rel_hf[c["consulta"]]
        filas.append([h["fecha"][:10], "Hugging Face Papers", "ad hoc", c["devueltos"], min(c["devueltos"], 12) if c["devueltos"] else 0,
                      r, "no", f"Consulta: {c['consulta']}. Revisados los 12 primeros. {o}"])
    for e in FILAS_EXTRA:
        filas.append(list(e))

    enc = ["fecha", "base", "cadena", "resultados", "revisados", "relevantes", "amenaza_C1_C2", "observacion"]
    with open(AQUI / "matriz_novedad.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(enc)
        w.writerows(filas)

    md = ["# Matriz de novedad (R1)", "",
          "Fecha de las consultas: 2026-11-23 (hora de Bogotá). Los conteos de resultados son los que declara cada fuente;",
          "los revisados son los títulos (y resúmenes, si los hay) efectivamente leídos, hasta 50 por consulta (12 en Hugging Face Papers);",
          "los relevantes siguen los criterios de `cadenas.md`. `amenaza` marca si algún relevante podría anticipar C1 (encogimiento atenuado) o C2 (expansión consistente):",
          "`parcial` significa que anticipa una parte (expansión del núcleo con offset regresado) sin llegar a la regla.", "",
          "| " + " | ".join(enc) + " |", "|" + "---|" * len(enc)]
    for f in filas:
        md.append("| " + " | ".join(str(x).replace("|", "/") for x in f) + " |")
    md += ["", "## Snowballing (OpenAlex y Semantic Scholar, 2026-11-23)", "",
           "Hacia atrás: referencias del trabajo; hacia adelante: citas, filtradas en el servidor por palabras del mecanismo",
           "(shrink, shrunk, kernel, unclip, expansion, offset, inscribed, thickness) en título o resumen para DBNet y DBNet++,",
           "y sin filtro para los demás. Crudos en `resultados/snowball__*.json` y `resultados/snowball_hazmat.json`.", "",
           "| semilla | referencias (resueltas) | citas totales | citas tras filtro | revisadas | relevantes | observación |",
           "|---|---|---|---|---|---|---|"]
    for clave, rel, obs in SNOW:
        d = json.loads((RES / f"snowball__{clave}.json").read_text(encoding="utf-8"))
        rev = len(d["adelante"])
        md.append(f"| {d['titulo_resuelto'][:70]} ({d['doi']}) | {d['referencias_declaradas']} ({d['referencias_resueltas']}) | "
                  f"{d['citas_totales']} | {d['citas_filtradas']} | {rev} | {rel} | {obs} |")
    sh = json.loads((RES / "snowball_hazmat.json").read_text(encoding="utf-8"))
    for s_ in sh["semillas"]:
        rel, obs = SNOW_HAZMAT[s_["semilla"]]
        md.append(f"| {s_['semilla']} (Semantic Scholar, DOI {s_['doi']}) | n/d | {s_['n_citas_devueltas']} (hasta 100) | sin filtro | {s_['n_citas_devueltas']} | {rel} | {obs} |")
    md += ["", "Las cifras entre fuentes no se suman como estudios únicos: un mismo trabajo aparece en varias.",
           "Los resultados crudos, con título, año, DOI o URL y resumen, están en `resultados/<fuente>__<cadena>.json`.", ""]
    (AQUI / "matriz_novedad.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(len(filas), "filas")


if __name__ == "__main__":
    main()
