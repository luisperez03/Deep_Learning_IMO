# Segmentación de etiquetas de mercancías peligrosas

**Deep Learning · Entregable 2 · Integración local del 20 de noviembre de 2026**

**Autores:** Luis Perez y Daniel Carbono. **Entrega prevista:** domingo 22 de noviembre.

## Resumen

Se compararon diez adaptaciones de redes para segmentación binaria de rótulos mediante cinco semillas por modelo. Se conservaron las particiones agrupadas del Entregable 1 y se entrenó desde cero a 256×256 píxeles, con selección del checkpoint por Dice de validación. La evaluación final comprendió 218 imágenes de 122 grupos de similitud. TransUNet alcanzó Dice 0,8626 y FCN compacta 0,8535. La diferencia pareada fue 0,0091, con IC95 de grupos [−0,0083; 0,0278], condicional a las cinco semillas; no establece mejora clara y el TOST con margen 0,02 no acredita equivalencia (p = 0,12). FCN ofrece una referencia compacta de 72.145 parámetros. Se auditaron las cincuenta corridas y las máscaras guardadas de semilla 11, y se añadieron análisis exploratorios corregidos por multiplicidad, curvas y comparación de costo. La interpretación reconoce exposición previa del test en E1, dependencia residual entre escenas y ausencia de imágenes totalmente negativas. La réplica de evaluación del paper guía en ADE20K dio 37,43 % mIoU frente al 37,4 % reportado, compatible con el criterio fijado antes de ejecutarla. Los diagnósticos sobre los pesos congelados (calibración, robustez, bordes, ensamble con predicción conforme y explicabilidad) se ejecutaron después como exploratorios: las redes compactas resultan mucho menos robustas a imágenes degradadas que las grandes. Lo que sigue pendiente se identifica expresamente en el checklist. Este documento integra evidencia ejecutada, sin convertir planes en resultados.

## Estado de entrega

El benchmark principal está cerrado y auditado. **La entrega integral todavía no está completa.** Consultar el [checklist requisito por requisito](checklist_e2.md) y el [plan de cierre](cierre_e2.md). El profesor acepta diez modelos totales, según confirmación del usuario; no se asume una exención de los demás requisitos.

```{tableofcontents}
```
