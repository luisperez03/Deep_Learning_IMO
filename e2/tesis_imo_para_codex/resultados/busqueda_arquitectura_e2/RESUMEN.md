# Búsqueda de arquitectura de las compactas: resumen automático

Solo validación; el test no se carga. Generado por `scripts/resumir_busqueda_arquitectura_e2.py`.

Ajuste previo (tasa elegida por familia): {'fcn': 0.001, 'unet': 0.0003, 'segnet': 0.0003}

| familia | n fase 1 | media | DE entre arq. | σ semilla | k | ganador | media fase 2 | IC95 | vs heredada |
|---|---|---|---|---|---|---|---|---|---|
| fcn | 1000 | 0.7226 | 0.0157 | 0.0105 | 10 | ((64, 64, 64), (0, 1, 0)) | 0.8508 | [0.8441, 0.8575] | el ganador supera a la heredada por menos de δ |
| unet | 1000 | 0.7286 | 0.0202 | 0.0134 | 10 | ((64, 64, 64), (0, 1, 0)) | 0.8729 | [0.8631, 0.8827] | el ganador supera a la heredada por δ o más |
| segnet | 1000 | 0.7207 | 0.0258 | 0.0143 | 10 | ((64, 64, 64), (0, 1, 0)) | 0.8607 | [0.8462, 0.8753] | el ganador supera a la heredada por δ o más |
