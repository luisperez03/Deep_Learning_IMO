# Colab E2 — paquete corregido v2

Usar **Benchmark_E2_Colab_v2.ipynb** junto a **paquete_colab_e2_v2.zip**, en la carpeta superior. El paquete antiguo `paquete_colab_e2.zip` y su carpeta extraída contienen prototipos y NO deben utilizarse como benchmark de las arquitecturas originales.

1. Abrir el notebook v2 en Colab y seleccionar GPU.
2. Subir el ZIP v2 al panel de archivos o a la raíz de Drive.
3. Ejecutar montaje, verificación del ZIP e instalación aislada (Python 3.11, torch 2.5.1/CUDA 12.1).
4. Ejecutar preflight. Un fallo bloquea entrenamiento del modelo; revisar el error registrado. VM-UNet exige el kernel CUDA real, nunca una GRU de reemplazo.
5. Ejecutar la celda de entrenamiento. Puede seleccionarse una parte de los modelos/semillas explícitamente. Guardado en una carpeta nueva de Drive, separado del piloto E1 y de prototipos.

## Qué se ejecuta

La fuente ejecutable es `config/experimento_e2.json`. Recetas fijas, inicialización aleatoria, máximo 30 épocas, paciencia 5, cinco semillas, tamaño 256, lote nominal 2. Último lote solitario se une al anterior (lote 3), sin duplicar ni perder imágenes. No hay búsquedas automáticas de hiperparámetros ni evaluación de test. Las tres bases se vuelven a correr: el E1 se conserva como histórico porque cambia el lote.

SMP 0.5.0 implementa DeepLabv3+/UNet++ ResNet-18 y SegFormer MiT-B0 con decoder de 256 canales. Sin pesos externos. UNet++ no incluye supervisión profunda del artículo. PIDNet-S y TransUNet R50-ViT-B/16 usan archivos de autores. SegNeXt conserva MSCAN-T y LightHamHead oficiales con integración PyTorch mínima: SyncBN→BN de un dispositivo, infraestructura MMCV/MMSeg reemplazada, bases NMF en el dispositivo del modelo. VM-UNet usa VSSM oficial con selective_scan de Mamba y logits antes de sigmoid. Commits/hashes/licencias: `third_party/e2/sources.lock.json` y carpetas correspondientes.

PIDNet usa pérdida principal BCE/Dice + 0,4 × auxiliar BCE/Dice + 1,0 × BCE de borde morfológico 3×3. Sólo zonas válidas y erosionadas para bordes. Es una adaptación binaria declarada, no la pérdida original multicategoría. Demás modelos: media por imagen de 0,5 BCEWithLogits + 0,5 soft Dice. Validación con igual peso por grupo; el azar NMF se fija durante evaluación.

## Persistencia y límites

Se verifica la firma de código, configuración, entorno, CSV y contenidos de imágenes/máscaras train/val antes de continuar. El test sólo aporta metadatos para comprobar separación de grupos. El ZIP no incluye sus imágenes/máscaras.

`last.pt` guarda modelo, optimizador, RNG Python/NumPy/torch/CUDA, historia y referencia al mejor checkpoint inmutable `best_epoch_XXX.pt`, con su validación. Al finalizar se crea `result.json`. Una caída a mitad de época repite esa época desde el último estado completo. No se continúa silenciosamente en almacenamiento temporal si falta Drive. Se requieren espacio suficiente y un solo proceso por corrida; no ejecutar dos notebooks simultáneamente sobre la misma carpeta.

Las ruedas Linux/CUDA de mamba-ssm 2.2.4 y causal-conv1d 1.5.0.post8 se obtienen de releases oficiales. Su existencia está verificada; la ejecución GPU se comprueba con preflight dentro de Colab. No se ha afirmado entrenamiento completo de las diez redes desde el portátil. El tiempo/memoria que mida preflight corresponde a un paso sintético, no a una época ni a presupuesto garantizado.

Este benchmark no satisface por sí solo la réplica de SegFormer en ADE20K, ni cierra cribado/PRISMA o aprobaciones docentes. Fuentes de literatura anteriores conservan observaciones de auditoría en `docs/revision_entregable2/REVISION_CAMBIOS_20261118.md`.
