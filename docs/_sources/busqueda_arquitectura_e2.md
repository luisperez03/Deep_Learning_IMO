# Búsqueda de arquitectura de las compactas

FCN, U-Net y SegNet son las tres redes del benchmark sin una arquitectura fijada por su artículo: se usaron las
versiones compactas heredadas del E1. La sección 14 de la guía pide, para esas redes, recorrer el espacio de
arquitecturas en dos fases. El protocolo se fijó el 22/11, antes de correr (`docs/PROTOCOLO_BUSQUEDA_ARQUITECTURA_E2.md`):

- **Espacio.** Tres bloques con anchos en {16, 32, 64, 128, 256} y dropout 0,2 encendido o apagado en cada bloque:
  1.000 arquitecturas por familia. La heredada, (16, 32, 64) sin dropout, está dentro.
- **Ajuste previo de la tasa.** Tres tasas con la arquitectura de referencia (64, 64, 64)/(0, 1, 0). La regla, fijada
  antes, se queda con la receta (0,001) salvo que otra tasa gane por 0,005 o más.
- **Fase 1, cribado.** Las 1.000 por familia con una semilla, a fidelidad reducida (128 px y 6 épocas, elegida por
  presupuesto con el piloto), más 40 repeticiones para medir el ruido de semilla.
- **Fase 2, confirmación.** Los 10 mejores de la fase 1, la referencia y la heredada, con las cinco semillas del
  benchmark, a 256 px y 30 épocas.
- **Selección.** Mayor media de validación. Si hay empate técnico con la mejor (diferencia menor que δ = 0,02 o
  intervalos al 95 % solapados), gana la de menos parámetros.

Todo se mide en validación. El test no se cargó: esta búsqueda es exploratoria y no cambia las conclusiones del
benchmark. Se ejecutó completa en Colab (`colab/Busqueda_Arquitectura_E2.ipynb`, L4); la fase 2 terminó el 26/11.

![Fase 1 frente a fase 2 (arriba) y parámetros frente a Dice de la fase 2 (abajo), por familia](figuras/e2/busqueda_arquitectura.png)

## Cifras por familia

```{include} ../resultados/busqueda_arquitectura_e2/TABLA_RESUMEN.md
:start-line: 2
```

```{include} ../resultados/busqueda_arquitectura_e2/LECTURA.md
:start-line: 2
```

## Detalle de la fase 2

```{include} ../resultados/busqueda_arquitectura_e2/TABLA_FASE2.md
:start-line: 2
```

Los datos de cada corrida (configuración, historia por época, predicciones de validación) están en
`resultados/busqueda_arquitectura_e2/`, sin los pesos, que quedan en Drive. Se regeneran las tablas, la lectura y la
figura con `scripts/lectura_busqueda_arquitectura_e2.py --figura`.
