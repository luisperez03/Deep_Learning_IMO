# Continuidad prioritaria — 20 de noviembre de 2026

Este documento prevalece sobre los estados históricos de CONTINUIDAD_E2.md. El E2 está avanzado, pero NO está completo. La entrega es el domingo 22/11/2026. No prometer una nota ni marcar pruebas preparadas como ejecutadas.

## Acuerdos confirmados

- El profesor acepta **10 redes totales**, incluyendo las tres de E1.
- Q5 y Q6 de búsqueda se omitieron por instrucción del usuario.
- El usuario tiene Colab con GPU y los 50 checkpoints en Drive.
- **No hay segundo anotador**. No fabricar acuerdo humano ni presentar Dice=1 o consenso de redes como techo empírico.
- El usuario autorizó completar experimentos complementarios. El test ya fue observado: los nuevos análisis son exploratorios, no preinscritos retrospectivamente.
- El 20/11 autorizó revisar GitHub y subir este avance para Daniel. Esta autorización no implica sobrescribir cambios ajenos. Para publicaciones futuras, consultar al usuario.

## Mapa local y remoto

Raíz local Windows: `C:\Users\luisa\Downloads\DEEP LEARNING`.

| Ruta relativa a la raíz local | Contenido |
|---|---|
| `tesis_imo_para_codex/` | Proyecto de trabajo, scripts, libro, protocolos y resultados derivados |
| `libro-imo-pages/` | Checkout GitHub: https://github.com/luisperez03/Deep_Learning_IMO |
| `tesis_imo_e2_arquitecturas_reales_v2/` | 50 corridas, best/last checkpoints, configuraciones y histories originales |
| `resultados_test_e2_final/` | Test cerrado, 50 JSON y 2.180 máscaras seed11 |
| `paquete_colab_e2_v2.zip` | Paquete válido de arquitecturas reales; NO usar el ZIP sin v2 |
| `Benchmark_E2_Colab_v2.ipynb` | Notebook original de entrenamiento |
| `paquete_test_e2_cerrado.zip`, `Evaluacion_Final_E2_Colab.ipynb` | Paquete/notebook de test congelado |
| `Entregables para la clase/Entregable2DeepLearningProject.pdf` | Guía completa de 65 páginas |
| `tmp/pdfs/entregable2_texto.txt` | Texto extraído de la guía, separadores PAGE |
| `Hazmat Placards.v5i.coco/`, `VOC2012/`, `SeaFront_v1_0_0/` | Datos originales: no editar ni publicar sin verificar licencia |

En GitHub el snapshot del proyecto está en `e2/tesis_imo_para_codex/`. Las rutas siguientes son relativas a ese proyecto. El snapshot NO lleva datasets, checkpoints, entornos virtuales, pesos clásicos ni imágenes originales. Daniel debe obtener los paquetes y pesos directamente de Luisa/Drive. Los datos se colocan como hermanos del proyecto según los paths de `resultados/preentrenamiento/*.csv`. No sustituir imágenes ni regenerar splits. El libro publicado previo permanece en `docs/`.

## Leer primero

1. `AGENTS.md` (la auditoría inicial ya fue realizada; no repetirla por su texto histórico).
2. `docs/revision_entregable2/CHECKLIST_ENTREGA_E2.md`: 76 controles y 21 casillas de la guía. Sus estados aún no incorporan todos los complementos nuevos.
3. `libro/metodos_e2.md`, `libro/resultados_e2.md`, `libro/cierre_e2.md`.
4. `config/protocolo_test_e2_cerrado.json` y `config/complementos_e2_20261120.json`.
5. `resultados/complementos_locales_e2/RESUMEN.json` y `resultados/baseline_clasico_e2/RESUMEN.json`.
6. `COMPLEMENTOS_CERRADOS_20261120.md` para los resultados nuevos.

## Evidencia terminada

- Benchmark: 10 modelos × 5 semillas, 50 corridas. Auditorías en `resultados/auditoria_e2_recibida/` y `resultados/auditoria_test_e2/`. 10.900 registros test y 2.180 máscaras seed11 verificados.
- TransUNet Dice test 0,8626045; FCN 0,8535431. Diferencia 0,0090614, IC95 bootstrap original [-0,0082835; 0,0277904]. No demuestra superioridad ni equivalencia ±0,02.
- Integración: `scripts/analisis_integracion_e2.py`, `integrar_entregable2.py`, `reproducir_entregable2.py`, `verificar_integracion_e2.py`. Libro y diagramas, curvas, comparación estadística y Pareto de parámetros existentes.
- Complementos locales **ejecutados**: `scripts/completar_analisis_locales_e2.py`; EDA de geometría/color, adversarial validation con 200 permutaciones de dominio, bordes seed11, descomposición descriptiva y TOST. Resultados en `resultados/complementos_locales_e2/`.
- Baseline clásico **terminado**: `scripts/baseline_clasico_e2.py`; HistGradientBoosting, 13 features, 3 configuraciones × 5 semillas, elección por validación. Test final Dice 0,6446858, DE semillas 0,0022226, IC95 grupos [0,6047324;0,6823762]. Resultados en `resultados/baseline_clasico_e2/`. Pesos `.joblib` y muestras `.npz` solo locales.
- Tests recientes: 3 tests de complementos y 6 de métricas de diagnóstico pasan. Runner y resumidor GPU compilan. Esto NO valida ejecución CUDA.

## Prioridad 1: cerrar el complemento Colab GPU

Archivos implementados pero **no ejecutados en GPU**:

- `scripts/metricas_diagnostico_e2.py`: Brier/NLL/ECE fijo y adaptativo, temperature scaling, bordes, perturbaciones y conjuntos de etiquetas.
- `scripts/diagnosticos_gpu_e2.py`: etapas `clean`, `robustness`, `interpret`; verifica hashes, firma de entorno y checkpoints, guarda JSON atómicos y reanuda por tarea completada.
- `scripts/resumir_diagnosticos_gpu_e2.py`: intervalos por grupos, gap train–val, curvas de robustez, ensembles y resumen de atribuciones.
- `tests/test_metricas_diagnostico_e2.py`, protocolo `config/complementos_e2_20261120.json`.

**Todavía falta empaquetar y crear el notebook complementario.** No decir que el notebook nuevo está listo. Usar como plantilla `scripts/preparar_test_e2.py` y el notebook de test existente; **NO ejecutar preparar_test_e2.py**, pues reescribe el protocolo cerrado. Construir un script nuevo que reutilice el ZIP cerrado, añada imágenes/máscaras train/val de los CSV (el ZIP de test solo lleva test), el protocolo complementario y los tres scripts. Preservar exactamente todos los archivos cubiertos por hashes del protocolo cerrado.

El entorno requerido es Python 3.11 aislado en `/content/imo-e2-env`, torch 2.5.1+cu121 y torchvision 0.20.1+cu121; reproducir el resto desde `training_environment.packages` del protocolo, incluyendo Mamba y causal-conv1d. Instalar `setuptools==75.8.0`. En una sesión nueva `/content` desaparece: recrear entorno y extraer datos antes de verificar. No usar el Python 3.13 del kernel para wheels cp311. Rutas CUDA deben incluir torch/lib y nvidia/*/lib del entorno.

Comandos orientativos dentro del paquete extraído (usar `-u` para logs visibles):

```bash
/content/imo-e2-env/bin/python -u scripts/diagnosticos_gpu_e2.py --runs /content/drive/MyDrive/tesis_imo_e2_arquitecturas_reales_v2 --output /content/drive/MyDrive/tesis_imo_e2_diagnosticos_v1 --stage clean
/content/imo-e2-env/bin/python -u scripts/diagnosticos_gpu_e2.py --runs /content/drive/MyDrive/tesis_imo_e2_arquitecturas_reales_v2 --output /content/drive/MyDrive/tesis_imo_e2_diagnosticos_v1 --stage robustness
/content/imo-e2-env/bin/python -u scripts/diagnosticos_gpu_e2.py --runs /content/drive/MyDrive/tesis_imo_e2_arquitecturas_reales_v2 --output /content/drive/MyDrive/tesis_imo_e2_diagnosticos_v1 --stage interpret
/content/imo-e2-env/bin/python -u scripts/resumir_diagnosticos_gpu_e2.py --output /content/drive/MyDrive/tesis_imo_e2_diagnosticos_v1
```

Revisar antes de entregar: perfiles CUDA/custom ops, conservación de hashes, recuperación tras interrupción, verificación de hashes NPZ también en resumidor, intervalos para fidelidad/estabilidad y cobertura por estratos. Comparar métricas clean con los JSON del test original: no asumir igualdad bit a bit al cambiar GPU. Los FLOPs del profiler son conteo parcial, no FLOPs certificados de Mamba/NMF. La cobertura de ensembles NO tiene garantía conformal. Top3 escogidos por validación: TransUNet, U-Net, FCN. Interpretabilidad usa 100 grupos de validación, 16 pasos IG, 5 semillas; categorías geométricas automáticas NO sustituyen revisión semántica humana. Robustez tiene 450 tareas (10×5×3 perturbaciones×3 niveles); medir costo con piloto, no inventar tiempo estimado.

## Prioridad 2: requisitos aún sin ejecución/implementación

- HPO con presupuesto equivalente entre redes; benchmark original tenía una receta por modelo. El clásico con tres candidatos no crea igualdad retrospectiva. Diseñar suplemento separado, seleccionar con val, documentar exposición previa al test.
- Tres ablaciones de componentes del mejor modelo por validación, cinco semillas. No implementadas; no confundir comparar arquitecturas distintas con ablación causal.
- Permutación de etiquetas del pipeline de segmentación (guía pide al menos 200 si costoso). Las 200 permutaciones locales son de **dominio train/test**, no satisfacen este control.
- Réplica del paper guía SegFormer en dataset/condiciones originales y análisis de desviación. No realizada. Antes de descargar nuevos datasets, aplicar aprobación exigida por AGENTS y verificar licencia/tamaño. No llamar réplica original al benchmark HAZMAT.
- Techo de desempeño: no segundo anotador. Investigar si hay una estimación de ruido/Bayes defendible con datos disponibles; si no, conservar el requisito como limitación explícita y consultar al profesor. Nunca inventar un techo humano.
- Revisión de fuga residual por embeddings; ejecución independiente en entorno limpio; defensa y material de presentación.

## Prioridad 3: bibliografía e integración final

La búsqueda tiene 13 exportaciones, 9.646 registros y 6.325 grupos provisionales. `cribado_automatico.json` NO es elegibilidad humana ni PRISMA final. Raw exports y `registros_conciliados.json` (73 MB) permanecen locales en `docs/revision_entregable2/`; obtenerlos de Luisa. No repetir búsquedas ni reabrir Q5/Q6.

Tres pares de DOI requieren resolver y aplicar aliases con evidencia editorial, preservando IDs originales:

- E2-3569/E2-5674: `10.7717/peerj-cs.2146` frente a `10.7717/peerjcs.2146`. PDF editorial https://peerj.com/articles/cs-2146.pdf confirma DOI con guion.
- E2-5639/E2-5682: `10.1504/ijes.2021.116134` frente a `10.1504/ijes.2021.10039166`.
- E2-5459/E2-5878: `10.3788/cjl202148.1507004` frente a `10.3788/cjl202118.1507001`.

`RESOLUCION_CONFLICTOS_DOI.md` tiene rectificación: la fusión NO está aplicada. Verificar fuentes primarias antes de transformar; si se confirman tres fusiones, serían 6.322 grupos, no 6.325. El parser actual `procesar_exportaciones.py` no aplica aliases. Falta cribado trazable de texto completo, tabla de resultados originales de papers, metadatos/cuartiles cuando proceda y síntesis de tres mecanismos transversales.

Integrar resultados nuevos y luego actualizar el checklist fila a fila con evidencia. No cambiar automáticamente todos los parciales a completos. `scripts/integrar_entregable2.py` sobrescribe index/modelos/resultados/reproducibilidad/TOC: editar el generador o usar un capítulo separado y añadirlo también al TOC generado. El capítulo `libro/complementos_e2.md` conserva este cierre local sin fingir diagnósticos GPU.

## Entornos y comprobaciones locales

- `.venv-e2/Scripts/python.exe`: torch CPU, útil para comprobar sintaxis; no VM-UNet CUDA.
- `.venv-complemento/Scripts/python.exe`: sklearn 1.6.1, scipy 1.15.3, numpy 2.2.6; complementos locales. Ver `requirements-complementos-locales.freeze.txt`.
- `.venv-libro/Scripts/jupyter-book.exe build libro -W --keep-going`: reconstruir libro. No usar `python -m jupyter_book`.
- Tests: `python -m unittest discover -s tests -p test_complementos_locales_e2.py` y `... -p test_metricas_diagnostico_e2.py`; integración: `test_integracion_e2.py`.
- Resultados científicos no cambian al recompilar. No correr de nuevo entrenamiento/test cerrado ni regenerar sus protocolos para arreglar enlaces.

## Transferencia a Daniel

La rama de avance contiene código/fuentes y evidencia numérica seleccionada, no una entrega final ni publicación automática. El manifiesto `e2/MANIFIESTO_SNAPSHOT.json` enumera archivos y SHA256. Verificar rama/remoto antes de mezclar con trabajo nuevo. No borrar `docs/` ni desplegar este avance como final. Priorizar notebook complementario para que Luisa ejecute GPU mientras Daniel cierra literatura y presentación. Pedir resultados exportados JSON, manteniendo checkpoints en Drive.
