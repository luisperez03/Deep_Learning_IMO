# Revisión de los cambios posteriores al traspaso

Fecha: 18/11/2026. Solicitud del usuario: revisar lo nuevo. Revisión estática de código, notebook, documentación y registros JSON; no se lanzó entrenamiento ni se modificaron implementaciones. Este informe actualiza el estado del traspaso anterior.

**Dictamen: no ejecutar todavía el benchmark como comparación de las diez arquitecturas anunciadas.** Hay trabajo aprovechable en documentación y orquestación, pero las identidades de cinco redes, el protocolo ejecutado y varias declaraciones de cierre necesitan corrección.

## Hallazgos prioritarios

### 1. P1 — Cinco nombres de arquitecturas corresponden a implementaciones diferentes

Evidencia: `scripts/modelos_e2.py`, clases desde líneas 89, 136, 164, 190 y 225.

- `SegFormerB0Adapter` contiene convoluciones y fusión multiescala; no bloques de atención Transformer ni MiT-B0.
- `VMUNetAdapter` usa `nn.GRU`; no implementa VSSM/Mamba ni el operador selectivo de VM-UNet. Ejecutar esta GRU en CPU no resuelve la compatibilidad del modelo original.
- `PIDNetAdapter` son tres ramas convolucionales concatenadas; no corresponde al PIDNet-S oficial y no entrega objetivo auxiliar de borde al runner.
- `SegNeXtAdapter` es un bloque convolucional de atención; no conserva el encoder MSCAN-T y LightHamHead documentados.
- `TransUNetAdapter` usa dos convoluciones y un Transformer de dos capas; no ResNet-50 + ViT-B/16 ni decoder con los skips propuestos. A 256 procesa una secuencia de 4096 tokens, por lo que tampoco debe presumirse menor coste por ser una simplificación.

Consecuencia: las métricas quedarían asociadas a modelos que no se entrenaron. Tampoco se acredita la cobertura Transformer/SSM anunciada. Una salida `(B,1,256,256)` sólo valida interfaz, no identidad arquitectónica.

Corrección: usar implementaciones fieles y versionadas, adaptando entrada/salida/pérdida. Si se conservan prototipos, darles nombres propios y tratarlos como experimentos diferentes; no cubrir con ellos las plazas originales.

### 2. P1 — Los fallbacks cambian el modelo silenciosamente

Evidencia: `modelos_e2.py:14` y `:63`. Ante ImportError, DeepLabv3+ se convierte en encoder + convolución 1×1 y UNet++ en SmallUNet + convolución final. El identificador de corrida permanece igual y `config.json` no registra `_mode`.

Consecuencia: dos entornos pueden generar resultados bajo el mismo nombre con arquitecturas distintas. Corrección: abortar si falta la dependencia requerida, fijar versiones y guardar identidad/configuración/hash del modelo. Los fallbacks de prueba deben estar fuera del registro del benchmark.

### 3. P1 — El entrenamiento no implementa la especificación

Evidencia: `experimento_e2_colab.py:124` instancia Adam para todos; el argumento global `--lr` vale 0,001 por defecto. El notebook no lo sobrescribe por modelo. La tabla de `ESPECIFICACION_EXPERIMENTO_E2.md:36` en adelante prescribe otros optimizadores y tasas (por ejemplo AdamW 0,00006 para SegFormer).

Además, `to_tensors` siempre agrega un lote de tamaño 1: no existe opción real para el lote físico 2 mencionado en el documento. No se implementan pérdidas auxiliares de PIDNet. La compatibilidad de BatchNorm en las implementaciones definitivas requiere una prueba de entrenamiento, no sólo inferencia.

Corrección: configuración explícita por modelo, sin alternativas ambiguas «AdamW / Adam», con pérdida, optimizador, tasa, lote y política de selección registrados. Verificar que el notebook consume esa configuración.

### 4. P1 — La reanudación no protege la identidad de datos/código

Evidencia: `experimento_e2_colab.py:117` compara sólo nombre, semilla, épocas, tasa, paciencia y tamaño. No guarda/verifica hash de manifiesto, normalización, máscaras ni implementación. Contradice la garantía sobre manifest de la especificación, línea 23. `main`, línea 262, omite cualquier corrida con `result.json` antes de validar configuración o su estado.

Consecuencia: una corrección del modelo, cambio de datos o cambio de presupuesto puede reutilizar resultados anteriores sin advertencia; una arquitectura distinta pero compatible en formas puede cargar pesos inadecuados.

Corrección: firma del experimento y validación antes de saltar/reanudar, incluyendo estado de finalización. Conservar los resultados antiguos como experimentos distintos.

### 5. P2 — Persistencia y reanudación se describen con garantías excesivas

Evidencia: `experimento_e2_colab.py:246` permite guardar en `/content/IMO_colab_e2_local` si Drive no está presente. No implementa la sincronización periódica mencionada en el diseño. Los checkpoints guardan RNG de torch/CUDA, no Python/NumPy; el cargador actual es determinista, pero no respalda una garantía general al añadir augmentations. `best.pt`, predicciones y `last.pt` se escriben por separado, sin transacción conjunta.

Consecuencia: el runtime puede perder resultados locales; cortes entre archivos pueden dejar mejor modelo/predicciones y estado de reanudación de épocas diferentes. `os.replace` por archivo no certifica persistencia remota ni consistencia del conjunto.

Corrección: exigir almacenamiento persistente cuando se promete persistencia, escribir un estado coherente/versionado y probar interrupciones en distintos puntos. Hablar de reanudación desde la última época completa; no prometer ausencia absoluta de repetición de cómputo.

### 6. P1 — El checklist declara pruebas/cierres que los artefactos no demuestran

Evidencia: `CHECKLIST.md:24` equipara cribado automático con título/resumen y texto completo; líneas 35–37 marcan confirmación docente, commits/hashes y medición de recursos como completados por existir especificación o forwards.

El notebook instala SMP sin versión. No hay en estos cambios una prueba de entrenamiento de las arquitecturas originales, ni medidas de RAM/VRAM/tiempo que resuelvan VM-UNet. La aprobación docente no se acredita mediante un documento generado; puede existir en otra conversación, pero falta registrar su evidencia.

Corrección: separar «documento preparado», «aprobado», «prueba ejecutada» y «resultado verificado». Reabrir las casillas sin evidencia. No interpretar el presente informe como revocación de autorizaciones que el usuario haya dado en otro chat.

### 7. P2 — PRISMA eleva propuestas automáticas a decisiones finales

Evidencia: `resumen_cribado.json` dice expresamente que requiere validación humana y no es PRISMA definitivo. `FLUJO_PRISMA.md:45` declara 2052 exclusiones como fin de evaluación; línea 84 convierte diez referencias priorizadas para lectura en estudios incluidos. `LECTURA_INICIAL_Q1.md` las describe como sugerencias desde títulos/resúmenes, sin lectura completa ni validación final.

Los conteos del JSON sí cuadran: 2186 retener + 2052 excluir + 2087 duda = 6325. Esto valida aritmética, no elegibilidad. El diagrama también pasa de registros a arquitecturas sin documentar el destino de los demás candidatos ni separar recuperación suplementaria.

Corrección: denominar las etiquetas precribado automático; adjudicar decisiones y documentar recuperados/no recuperados, texto completo y razones antes de cerrar PRISMA. Conservar selección operativa de siete redes separada del flujo exhaustivo.

### 8. P2 — La resolución DOI no está aplicada y el número final es contradictorio

Comprobación directa en `registros_conciliados.json`: persisten seis IDs para las tres parejas:

| Pareja | IDs existentes |
|---|---|
| PeerJ | E2-3569 / E2-5674 |
| Inderscience | E2-5639 / E2-5682 |
| Chinese Journal of Lasers | E2-5459 / E2-5878 |

`conciliacion.json` aún contiene tres ambigüedades. El cierre de `RESOLUCION_CONFLICTOS_DOI.md` afirma que ya estaban agrupadas, pero el JSON y el parser muestran lo contrario. `FLUJO_PRISMA.md:30` ofrece «6322 o 6325», que no constituye un conteo consolidado.

Corrección: verificar cada identidad con referencias directas y guardar decisiones de fusión con alias de IDs. Si se confirman las tres fusiones independientes, quedan 6322 estudios/grupos activos y 3324 duplicados conciliados, conservando 6325 IDs históricos sólo para trazabilidad. Recalcular cribado desde la correspondencia final. Esta revisión no verificó externamente que los DOI descartados sean inválidos: la explicación editorial del nuevo documento necesita URLs de evidencia.

### 9. P2 — Puntuación y comparación causal necesitan rectificación

Evidencia: `PUNTUACION_Y_EXCLUSIONES.md:45` dice que U-Net del E1 comparte ResNet-18 con UNet++; `scripts/modelos_base.py:29` muestra SmallUNet propia de canales 16/32/64. No se puede atribuir la diferencia sólo a conexiones densas.

PIDNet recibe pertinencia 4 por bordes, aunque la rúbrica exige evidencia en rótulos/dominio comparable para 4; la justificación presentada no aporta esa evidencia. El subtotal sin citas puede mostrarse como sensibilidad, pero el protocolo mantiene el total final pendiente, no aprobado. «Código y pesos verificados» debe diferenciar disponibilidad de comprobación de descarga/integridad.

Corrección: ajustar justificación/nota por componente, mantener puntuación final pendiente, y describir comparaciones entre modelos completos cuando encoder/capacidad también cambian.

## Trabajo aprovechable y verificación realizada

- Se añadieron fichas, documentación experimental, un capítulo del libro y artefactos HTML/PDF, además de notebook, empaquetador y runner. La existencia del build no demuestra que su contenido científico sea correcto.
- El runner conserva train/val separados, usa la máscara válida en BCE/Dice y selecciona por promedio de grupos. Tiene guardado por época y base útil para reanudación.
- Los conteos de las ocho etiquetas de precribado fueron recalculados desde el JSON y coinciden con el resumen.
- Se compararon SHA256 del runner y adaptadores entre proyecto y carpeta descomprimida `paquete_colab_e2`: son idénticos; los problemas también están en esa copia. No se inspeccionó el contenido interno del ZIP para certificar su equivalencia.
- No se ejecutaron las 50 corridas, no se instalaron dependencias ni se accedió a resultados remotos de Colab. No se certifica aquí la exactitud de todas las cifras bibliográficas nuevas ni la maquetación del PDF.

## Orden recomendado de corrección

1. Corregir identidad de las redes y eliminar sustituciones silenciosas; impedir que prototipos generen resultados con nombres oficiales.
2. Alinear runner/notebook con configuración explícita y agregar firma de datos/código y controles de persistencia.
3. Validar con pruebas sintéticas de forward/backward y reanudación antes de cualquier entrenamiento completo.
4. Rectificar checklist, puntuación, DOI y PRISMA; regenerar libro/informe/paquete sólo después de corregir fuentes.
5. Actualizar contexto de continuidad y conservar la instrucción del usuario: pedir autorización antes de subir a GitHub o desplegar.

No se han corregido silenciosamente los archivos revisados: este informe deja los hallazgos concretos para la siguiente intervención.
