(sec-archivo-autoria)=
# Archivo, autoría y anexos

## Archivo versionado del repositorio

El código, el protocolo, el libro y las tablas de este entregable se archivan como la *release* `v1.0.0-e3` del
repositorio [luisperez03/Deep_Learning_IMO](https://github.com/luisperez03/Deep_Learning_IMO), con licencia MIT
(`e3/LICENSE`) y metadatos de cita en `e3/CITATION.cff` y `.zenodo.json`. Zenodo asigna el identificador persistente
al publicar la *release*.

```{admonition} Identificador persistente
:class: important
DOI: **pendiente de asignación por Zenodo** al publicar la *release* `v1.0.0-e3`. Cuando exista, se escribe aquí y
en `e3/CITATION.cff`.
```

Cita recomendada: Carbonó Domínguez, D. y Pérez, L. (2026). *Deep_Learning_IMO, Entregable 3: UNet-AT y benchmark de
segmentación de rótulos de mercancías peligrosas (HAZMAT-13)* (versión 1.0.0-e3) [software]. Zenodo.

Componentes publicados aparte:

| Componente | Dirección | Contenido |
|---|---|---|
| Dashboard | <https://hazmat13-e3.onrender.com> | Aplicación Dash en Render, plan gratuito |
| Repositorio del dashboard | <https://github.com/ddcarbono/hazmat13-e3> | Código, tablas, artefactos ONNX de FCN y U-Net a0498, Dockerfile, CI y prueba de humo |
| Libro HTML | `docs/e3/` de este repositorio | Versión compilada de este libro |

Desviación declarada respecto del protocolo (§11): el protocolo preveía servir los artefactos ONNX desde un almacén
privado. Se publicaron en el repositorio del dashboard porque son pesos propios entrenados desde cero, sin
preentrenamiento, y no contienen imágenes del conjunto; las fotografías y máscaras de HAZMAT-13 siguen sin
redistribuirse.

## Declaración de autoría, contribuciones y uso de herramientas

El trabajo es de los dos integrantes. La tabla resume la contribución principal de cada uno por componente; ambos
revisaron el conjunto y responden por él en la sustentación.

| Componente | Daniel Carbonó Domínguez | Luis Pérez |
|---|---|---|
| Diseño del mecanismo (C1, C2), bitácora de diseño y demostración | Principal | Revisión |
| Estándar, protocolo congelado y análisis de potencia | Principal | Revisión |
| Implementación del original (`src/original_model.py`, `src/geometria_at.py`, entrenamiento) y pruebas | Primera versión | Integración, resolución de la discrepancia geométrica (OpenCV 4.11) y reanudación verificada |
| Revisión de novedad | Principal | Revisión |
| Búsqueda de arquitectura de las compactas (a0498) y su campaña en Colab | Principal | Revisión |
| Secuencia v4 de Colab, campaña del original y de los rivales, auditoría | Revisión | Principal |
| Integración de compactas, exportación ONNX y TorchScript, comparador de test | Revisión | Principal |
| Dashboard publicado (Render, pestañas de Modelos, explicación por oclusión, CI, prueba de humo) | Principal | Primera versión del tablero |
| Libro E3 | Capítulos de método, protocolo y cierre | Estructura, integración de resultados y revisión editorial |
| Reproducción independiente y congelamiento verificado | Principal | Revisión |

**Uso de herramientas, incluidas las de inteligencia artificial.** Se usaron asistentes de IA para escribir y revisar
código, redactar borradores y verificar consistencia: Claude Code (Anthropic) y asistentes de OpenAI (ChatGPT y
Codex). Cada resultado que se reporta sale de un archivo versionado y ejecutable (scripts, cuadernos y pruebas), no
de una respuesta del asistente; las cifras se recalculan con `experiments/reproduccion_independiente.py`. Las
decisiones de diseño, el protocolo y la interpretación son de los autores. Otras herramientas: PyTorch, ONNX Runtime,
Dash, Jupyter Book, Google Colab (GPU NVIDIA L4), GitHub Actions y Render.

```{list-table} Firmas
:header-rows: 1

* - Integrante
  - Firma
  - Fecha
* - Daniel Carbonó Domínguez
  - &nbsp;
  - &nbsp;
* - Luis Pérez
  - &nbsp;
  - &nbsp;
```

(sec-anexo-reproduccion)=
## Anexo A. Reproducción independiente

`experiments/reproduccion_independiente.py` regenera, en una carpeta temporal y sin GPU, las cifras publicadas a
partir de sus fuentes, sin leer las tablas que compara. Reporte: `experiments/reproduccion_independiente.json`
(2026-11-28; Python 3.13, numpy 2, pandas 2).

| Comprobación | Resultado |
|---|---|
| `generar_tabla_maestra.py` desde los resultados por imagen del E2 (el generador falla si una media no reproduce el informe del E2 a 4 decimales) | Aprobada |
| Tabla regenerada frente a `data/tabla_maestra.csv` de este repositorio: 11 filas × 14 columnas numéricas, tolerancia 5e-5 | Sin diferencias |
| Tabla regenerada frente a la del dashboard publicado | Sin diferencias |
| Media, DE y diferencia pareada del original recalculadas desde las 29 corridas de `data/avance_colab_corridas.csv` (20 del original y 9 de SegFormer) | Sin diferencias |
| Documentos congelados (`verificar_congelamiento.py`) frente a la etiqueta `e3-protocolo-v2` | Protocolo, procedencia y bitácora idénticos; estándar idéntico salvo la nota de auditoría del 27/11 |

Además, la integración continua corre en cada cambio, desde un clon limpio: las pruebas del original, de la secuencia y
del orquestador en este repositorio (`.github/workflows/e3.yml`), y en el del dashboard las pruebas, la construcción
del contenedor Docker y la prueba de humo contra el contenedor.

Alcance: es una reproducción automatizada desde un entorno limpio, no una reproducción por una persona ajena al
equipo. No reentrena modelos (requiere GPU; las identidades y huellas de cada corrida las audita
`experiments/auditar_campana.py`) ni evalúa el test del original, que no se ha evaluado.

(sec-anexo-humo)=
## Anexo B. Prueba de humo del despliegue

`scripts/prueba_humo.py` del repositorio del dashboard (solo biblioteca estándar) contra
<https://hazmat13-e3.onrender.com>, el 2026-11-28 a las 04:28 UTC. Reporte: `experiments/prueba_humo_render.json`.

| Comprobación | Resultado | Segundos |
|---|---|---|
| Páginas Contexto, EDA y Modelos | 200 | 0,4 a 1,0 |
| Layout y lista de callbacks de Dash | Servidos (11 callbacks) | 0,9 |
| Cabeceras de seguridad (nosniff, DENY, no-referrer) | Presentes | 0,5 |
| Predicción en vivo FCN a0498 con una imagen sintética | Máscara recibida | 1,8 |
| Predicción en vivo U-Net a0498 con una imagen sintética | Máscara recibida | 3,2 |
| Archivo que no es imagen | Rechazado con mensaje, sin error del servidor | 0,5 |
| Explicación por oclusión, FCN (parches de 32 px) | Mapa recibido | 34,9 |
| Explicación por oclusión, U-Net (parches de 64 px) | Mapa recibido | 40,9 |

Resultado: 10 de 10. La imagen de prueba se genera en el propio script; no se usan fotografías del conjunto. En el plan
gratuito de Render, la primera visita tras un periodo sin uso tarda de 30 a 60 s en despertar el servicio.
