# Diagnósticos sobre los pesos congelados

**Exploratorio, posterior a la exposición del test.** Se evaluaron de nuevo los 50 checkpoints cerrados (10 redes ×
5 semillas), sin reentrenar ni cambiar umbral, modelo o checkpoint. Corrida en una GPU L4 con
`colab/Complemento_GPU_E2.ipynb` y `scripts/diagnosticos_gpu_e2.py`; resumen con `scripts/resumir_diagnosticos_gpu_e2.py`
y tablas con `scripts/leer_diagnosticos_gpu_e2.py`. Salidas en `resultados/diagnosticos_gpu_e2/`.

**Control de identidad.** La etapa `clean` reproduce el Dice de test cerrado de las diez redes con diferencia máxima
menor que 0,0001: los pesos evaluados son los mismos del benchmark.

## Lo que muestran

**Ajuste a los datos.** El Dice de train, medido en modo evaluación con el mismo checkpoint, supera al de validación
en las diez redes; en SegNet la brecha (0,026) tiene un IC que incluye el cero. Es moderada en FCN, U-Net y DeepLabv3+ (≈ 0,05) y grande en SegNeXt-T (0,147
[0,110; 0,186]) y SegFormer-B0 (0,213 [0,166; 0,260]). El último lugar de SegFormer no es falta de capacidad para
ajustar: memoriza train y generaliza peor, lo que es coherente con entrenarlo desde cero con 1.015 imágenes. La
validación ya se usó para elegir checkpoints, así que la brecha es diagnóstica, no una prueba.

**Calibración.** El ECE con 15 intervalos queda entre 0,008 y 0,014 en todas las redes. Es bajo en buena parte
porque el fondo domina los píxeles y es fácil. La temperatura ajustada en validación es mayor que 1 en todas las
redes y semillas (1,6 a 2,6): los logits están sobreconfiados. Dividirlos por esa temperatura baja el Brier en todas
las redes, pero sube levemente el ECE15. Las dos métricas miden cosas distintas y no se elige entre ellas con el test.

**Robustez.** El resultado más claro. Con ruido y desenfoque, las redes grandes conservan entre el 92 y el 96 % del
Dice limpio (TransUNet, VM-UNet, DeepLabv3+, UNet++, PIDNet-S), y con cambios de brillo entre el 84 y el 98 %. Las
tres compactas del E1 conservan entre el 68 y el 77 % con las tres perturbaciones (FCN, U-Net y SegNet; 72 mil a
118 mil parámetros). El argumento de
parsimonia a favor de FCN vale con imágenes limpias, pero no con imágenes degradadas: ahí la diferencia con TransUNet
es grande y su IC no se cruza. SegFormer-B0 mejora levemente con desenfoque (1,018 [1,006; 1,033]).

**Bordes.** TransUNet tiene el menor HD95 (17,1 píxeles) y SegFormer-B0 el mayor (34,3), con el recall más bajo
(0,688): pierde partes del rótulo más de lo que inventa.

**Ensamble e incertidumbre.** Promediar las cinco semillas sube el Dice entre 0,021 y 0,028 en las tres mejores
redes, con IC que excluye el cero; el ensamble de TransUNet llega a 0,883. La predicción conforme con un umbral por
clase, calibrada en validación para una cobertura de 0,90, la cumple en promedio en test (0,903 a 0,912), pero entre
el 34 y el 37 % de los grupos queda por debajo: la garantía es marginal, no por imagen.

**Explicabilidad.** En las tres redes, borrar los píxeles que Integrated Gradients marca como importantes baja la
predicción más que borrar píxeles al azar, en el 89 % de los casos para TransUNet y en el 100 % para U-Net y FCN. Los
mapas son estables entre semillas en U-Net y FCN (Spearman ≈ 0,92) y menos en TransUNet (0,73). La ventaja está en
unidades de logit, que no se comparan entre redes.

## Tablas

```{include} ../resultados/diagnosticos_gpu_e2/TABLAS.md
:start-line: 4
```
