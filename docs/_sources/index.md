# Segmentación de etiquetas de mercancías peligrosas

**Deep Learning · Entregable 2 · Versión revisada del 26 de noviembre de 2026**

**Autores:** Luis Perez y Daniel Carbono. **Cierre experimental:** 26 de noviembre de 2026.

## Resumen

Se compararon diez adaptaciones de redes para segmentación binaria de rótulos mediante cinco semillas por modelo. Se conservaron las particiones agrupadas del Entregable 1 y se entrenó desde cero a 256×256 píxeles, con selección del checkpoint por Dice de validación. La evaluación final comprendió 218 imágenes de 122 grupos de similitud. TransUNet alcanzó Dice 0,8626 y FCN compacta 0,8535. La diferencia pareada fue 0,0091, con IC95 de grupos [−0,0083; 0,0278], condicional a las cinco semillas; no establece mejora clara y el TOST con margen 0,02 no acredita equivalencia (p = 0,12). FCN ofrece una referencia compacta de 72.145 parámetros. Se auditaron las cincuenta corridas y las máscaras guardadas de semilla 11, y se añadieron análisis exploratorios corregidos por multiplicidad, curvas y comparación de costo. La interpretación reconoce exposición previa del test en E1, dependencia residual entre escenas y ausencia de imágenes totalmente negativas. La réplica de evaluación del paper guía en ADE20K dio 37,43 % mIoU frente al 37,4 % reportado, compatible con el criterio fijado antes de ejecutarla. Los diagnósticos sobre los pesos congelados (calibración, robustez, bordes, ensamble con predicción conforme y explicabilidad) se ejecutaron después como exploratorios: las redes compactas resultan mucho menos robustas a imágenes degradadas que las grandes. El checklist detalla los pendientes documentales y de sustentación.

## Estado de entrega

El benchmark y los complementos experimentales están documentados. Quedan pendientes el ensayo de sustentación y el registro individual de la revisión humana de errores. El [checklist de entrega](checklist_e2.md) distingue estos pendientes de las limitaciones históricas; la [discusión](cierre_e2.md) recoge sus implicaciones. El alcance acordado con el docente comprende diez modelos: tres del E1 y siete adicionales.

```{tableofcontents}
```
