# Equipo y presupuesto propuesto antes de entrenar

Consulta de solo lectura realizada el 6 de noviembre de 2026. Valores en
`resultados/preentrenamiento/hardware.json`. No se ejecutó una prueba de carga.

| Recurso | Evidencia local |
|---|---|
| Equipo | ASUS Vivobook Go E1504FA |
| CPU | Ryzen 5 7520U, 4 núcleos / 8 hilos |
| RAM visible para Windows | 7,28 GiB (equipo de la clase comercial de 8 GB) |
| RAM libre durante la consulta | Aproximadamente 1,00 GiB; cambia con las aplicaciones abiertas |
| GPU reportada | AMD Radeon integrada; AdapterRAM reporta 512 MiB |
| Disco libre | Aproximadamente 272 GiB en la consulta |
| Framework de entrenamiento | PyTorch no está instalado en el Python usado para el EDA |

AMD identifica la GPU integrada de este procesador como Radeon 610M. Los 512 MiB
reportados por Windows no son toda la memoria compartida ni un presupuesto de VRAM
validado. Fuente: [especificación del procesador](https://www.amd.com/en/products/processors/laptop/ryzen/7000-series/amd-ryzen-5-7520u.html).

## Perfil inicial propuesto, aún sin ejecutar

- CPU; no asumir aceleración CUDA en esta GPU AMD. Cualquier backend AMD requiere
  una validación separada. No se instalaron controladores ni paquetes de GPU.
- Dos hilos de cómputo, cero procesos auxiliares de carga, batch 1, una ejecución
  a la vez, sin cachear todas las imágenes en RAM.
- Entrada candidata 256×256 con relación de aspecto conservada y padding ignorado
  por la pérdida. Revisar diagnóstico de resolución calculado solo sobre train.
- Comenzar posteriormente con una prueba autorizada de 8 imágenes y hasta 5 pasos;
  medir RAM máxima y tiempo antes de proyectar épocas o pilotos.
- Comprobar al menos 2 GiB de RAM libre antes de esa prueba. Es una reserva operativa
  propuesta, no una garantía de que cualquier arquitectura quepa.

Actualmente no hay evidencia de que el presupuesto de entrenamiento quepa: pesos,
activaciones, gradientes y estados del optimizador dependen de la arquitectura final.
No se inventan tiempos por época ni temperaturas. No se midieron sensores térmicos.
No ejecutar simultáneamente varios modelos ni pilotos completos sin medir primero.

## Entorno

Mantener el entorno del EDA separado del futuro entorno de entrenamiento. Seleccionar
la instalación CPU y fijar versiones después de aprobarla; guardar `pip freeze`,
Python y configuración de hilos del entorno real. La instalación se deberá contrastar
con [las instrucciones oficiales de PyTorch](https://pytorch.org/get-started/locally/).
No se descargó PyTorch ni se crearon modelos en esta etapa.

`scripts/diagnostico_pc.ps1` permite repetir la consulta CIM. `scripts/preflight.py`
comprueba memoria libre actual mediante Windows, integridad y aprobaciones, sin
iniciar cómputo neuronal. El indicador training_allowed permanece falso.
