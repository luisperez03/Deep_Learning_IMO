# Reproducción y archivos de entrega

## Compilar el libro después de editar el texto

Desde la raíz del proyecto, con las dependencias del libro instaladas:

```powershell
jupyter-book build libro -W --keep-going
```

La configuración mantiene desactivada la ejecución de notebooks. Este paso genera el HTML a partir de los textos, figuras y tablas existentes; no entrena modelos ni recalcula los análisis.

## Regenerar análisis y figuras desde los resultados

Desde la raíz `tesis_imo_para_codex`, con el entorno `.venv-libro` ya instalado:

```powershell
.\.venv-libro\Scripts\python.exe scripts\reproducir_entregable2.py
```

Este comando regenera los análisis exploratorios y las figuras, ejecuta las pruebas del análisis y compila el libro con advertencias tratadas como errores. Los textos generados por la integración se guardan en `resultados/integracion_e2/libro_generado/`; por defecto no sustituyen los capítulos editados de `libro/`. Lee los resultados originales en las carpetas hermanas `resultados_test_e2_final` y `tesis_imo_e2_arquitecturas_reales_v2`, los manifiestos y datos locales. No entrena ni publica. La reproducción del informe no es una repetición independiente de inferencia.

El resultado se abre en `libro/_build/html/index.html`. Dependencias exactas del informe en {download}`requirements-libro.txt <../requirements-libro.txt>`. El comando supone la estructura local y datos disponibles; no descarga silenciosamente datasets.

## Reproducción experimental

Los notebooks `Benchmark_E2_Colab_v2.ipynb` y `Evaluacion_Final_E2_Colab.ipynb` están en `colab/`; sus instrucciones identifican los paquetes y resultados externos necesarios. El paquete antiguo de prototipos no describe este benchmark. Los pesos se conservan en las cincuenta carpetas originales y no se incluyen en el libro HTML.

La normalización y geometría se fijaron con train; cada corrida selecciona por validación. Test E2 se ejecutó con checkpoints y umbral fijados. Repetir entrenamientos/inferencia debe ser una decisión explícita de reproducción; no es necesario para recompilar el documento.

## Evidencia descargable

- {download}`Configuración aplicada <../config/experimento_e2.json>`
- {download}`Protocolo cerrado de test <../config/protocolo_test_e2_cerrado.json>`
- {download}`Fuentes de autores y hashes <../third_party/e2/sources.lock.json>`
- {download}`Auditoría del entrenamiento <../resultados/auditoria_e2_recibida/auditoria.json>`
- {download}`Auditoría del test <../resultados/auditoria_test_e2/AUDITORIA.json>`
- {download}`Análisis exploratorio completo <../resultados/integracion_e2/analisis.json>`
- {download}`Código del análisis <../scripts/analisis_integracion_e2.py>`
- {download}`Código de auditoría de test <../scripts/auditar_test_e2.py>`
- {download}`Pruebas de la integración <../tests/test_integracion_e2.py>`
- {download}`Checklist de entrega <../docs/revision_entregable2/CHECKLIST_ENTREGA_E2.md>`
- {download}`Referencias BibTeX de las diez arquitecturas <referencias_e2.bib>`

La reproducción del 22/11 en un clon limpio del commit `7b669a7` obtuvo discrepancia cero en las tablas de test y diferencias de redondeo en los análisis. El recálculo desde 2.180 máscaras también coincidió. La reinferencia GPU de la etapa `clean`, realizada por separado, difirió como máximo en 1,3 × 10⁻⁵ de Dice medio por red. No se repitió todo el entrenamiento: dos figuras de curvas se conservaron porque las historias estaban en Drive. El detalle está en `resultados/reproduccion_completa_e2/LECTURA.md` y las instrucciones en `docs/REPRODUCCION_EXTREMO_A_EXTREMO.md`.

El libro se publica en GitHub Pages desde `docs/`. La publicación documentada más reciente es del 26/11; esta revisión editorial actualiza esa versión. La licencia de las imágenes sigue sin certificarse y limita su redistribución y la de los pesos.
