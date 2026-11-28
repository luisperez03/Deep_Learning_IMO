(sec-ablacion)=
# Ablación

La ablación es factorial completa 2³ sobre C1, C2a y C2b (`diseno.md` §5.1): ocho variantes, de las cuales solo dos se
entrenan (`dbnet` y `dbnet_at`); las otras seis salen de los checkpoints cambiando únicamente la inferencia
(`experiments/protocolo.md`, §8). No cambia ni parámetros ni FLOPs, así que no aplica control de capacidad.

**Diseño de las pruebas.** Cada variante contra el modelo completo, con las tres pruebas y Holm dentro de la familia
(m = 7), en los 122 grupos y en el tercil chico (dos familias de 7). Efectos principales e interacciones con IC 95 %
bootstrap por grupos.

**Predicciones declaradas antes de correr** (se reportarán cumplidas o fallidas): casi toda la ganancia del tercil
chico viene de C1; el HD95 del tercil chico baja; la ventaja desaparece al quitar C1 y no aparece con C1 sin C2a.

```{admonition} Pendiente
:class: warning
- **Tabla de la ablación (8 variantes, dos familias):** requiere los checkpoints de los dos brazos entrenados con 10
  semillas; saldrá del script de evaluación de variantes (por escribir en `experiments/`, usando
  `training/entrenar_original.py` y `src/geometria_at.py`).
- **Dice de validación de las 6 variantes por inferencia:** `MODELO_ORIGINAL_ESTADO.md` lo lista como no hecho.
```
