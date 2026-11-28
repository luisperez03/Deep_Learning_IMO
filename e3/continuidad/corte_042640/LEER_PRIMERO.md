# Corte 20261128T042640Z — continuación para Daniel

Este es el corte más reciente recibido. Sustituye el estado de avance del paquete `013803`,
pero no modifica ni elimina ese respaldo.

**Usar el lanzador corregido:** [reanudación exacta de fuentes y logs en Drive](REANUDACION_CORREGIDA.md).
Incluye una celda para cargarlo en el cuaderno actual sin subir otro notebook.

## Estado verificado

- Original: 20 corridas completas, diez por brazo.
- SegFormer corregido: 10/10 completas, incluida la semilla 107.
- Compactas a0498: 15 corridas nuevas completas, cinco por familia. Junto a las quince recuperadas del E2, completan diez semillas por familia.
- Auditoría con los resultados reutilizables del E2: **90/120 verificadas**.
- Rivales A: faltan cinco semillas de TransUNet y cinco de PIDNet. El progreso recibido señala etapa 06 ejecutándose, sin resultados completos incluidos.
- Rivales B: faltan cinco semillas de VM-UNet, SegNeXt, DeepLabV3+ y U-Net++ (20 corridas). No están terminadas según la confirmación de Luisa.
- HPO del original (04), evaluación y cierre científico siguen pendientes; no están incluidos entre las 120 corridas principales.

No volver a entrenar original, SegFormer ni compactas completadas. No declarar lista la evaluación integral de 120 corridas.

## Recuperar el contenido completo del ZIP recibido

Para evitar volver a transferir cientos de MB idénticos, el nuevo release contiene una actualización
con los archivos nuevos o modificados. El manifiesto enumera **todos** los archivos del corte recibido,
su SHA-256 y si proceden del paquete anterior o de la actualización. La reconstrucción produce el mismo
contenido de archivos del ZIP recibido, aunque no recrea su contenedor ZIP byte a byte.

Descargar:

1. [ZIP base 013803](https://github.com/luisperez03/Deep_Learning_IMO/releases/download/e3-avance-20261127/tesis_imo_e3_secuencia_v4-20261128T013803Z-1-001.zip), si no se conserva ya.
2. [Actualización y herramientas del corte 042640](https://github.com/luisperez03/Deep_Learning_IMO/releases/tag/e3-avance-20261128-042640): `actualizacion_e3_042640.zip`, `reconstruir.py` y `manifiesto_reconstruccion.json`.

Guardar los archivos juntos y ejecutar con Python estándar, sin instalar dependencias:

```text
python reconstruir.py --base tesis_imo_e3_secuencia_v4-20261128T013803Z-1-001.zip --actualizacion actualizacion_e3_042640.zip --salida corte_042640
```

Si el ZIP anterior ya está extraído, `--base` acepta la carpeta que contiene `tesis_imo_e3_secuencia_v4`.
La salida debe ser nueva: no se sobrescribe una campaña activa ni un paquete más reciente.
Todos los archivos se verifican antes de escribir. Los registros de auditoría están además en `registros/`.

## Continuar entrenamiento

Si la sesión de Luisa sigue ejecutándose, dejarla continuar: no correr en otra máquina el mismo carril ni copiar
archivos encima de sus checkpoints. Para un relevo tras detenerse esa sesión, usar el lanzador único v4 de
`e3/continuidad/` y el mismo destino de Drive. El carril `todo` verifica y conserva 01–03, 08 y 05;
continúa con 06, 07, 04, 09 y 10. Revisar la identidad de cualquier corrida parcial antes de cambiar GPU o entorno.

Cuando termine A o B, entregar un nuevo corte para integrar los resultados. El libro y el tablero publicados
en el corte anterior no se convierten automáticamente en una evaluación final por subir estos checkpoints.
No mezclar la tabla histórica de test con las nuevas métricas de validación.
