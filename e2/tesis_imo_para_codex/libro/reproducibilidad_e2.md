# Reproducción y archivos de entrega

## Regenerar el informe local

Desde la raíz `tesis_imo_para_codex`, con el entorno `.venv-libro` ya instalado:

```powershell
.\.venv-libro\Scripts\python.exe scripts\reproducir_entregable2.py
```

Este comando regenera análisis exploratorios, figuras, capítulos e índice, ejecuta las pruebas del análisis y compila el libro con advertencias tratadas como errores. Lee los resultados originales en las carpetas hermanas `resultados_test_e2_final` y `tesis_imo_e2_arquitecturas_reales_v2`, los manifiestos y datos locales. No entrena ni publica. La reproducción del informe no es una repetición independiente de inferencia.

El resultado se abre en `libro/_build/html/index.html`. Dependencias exactas del informe en {download}`requirements-libro.txt <../requirements-libro.txt>`. El comando supone la estructura local y datos disponibles; no descarga silenciosamente datasets.

## Reproducción experimental

Los notebooks de entrenamiento v2 y test cerrado están en la carpeta superior: `Benchmark_E2_Colab_v2.ipynb` y `Evaluacion_Final_E2_Colab.ipynb`, junto con sus ZIP versionados. El paquete antiguo de prototipos no describe este benchmark. Los pesos se conservan en las cincuenta carpetas originales y no se incluyen en el libro HTML.

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

Las auditorías verificaron la consistencia de los artefactos recibidos. Falta una repetición experimental independiente en entorno limpio que informe discrepancia de inferencia, no solo de agregación aritmética. Los originales, fotografías y pesos no deben publicarse sin revisar licencias. No se realizó push ni despliegue; cualquier publicación requiere autorización del usuario.
