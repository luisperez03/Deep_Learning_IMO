# Ficha del modelo: UNet-AT

> **Trazabilidad.** Ficha revisada el 2026-11-27; configuración original recuperada del 2026-11-23. Fuentes: `experiments/protocolo.md`, `experiments/estandar.md`,
> `MODELO_ORIGINAL_ESTADO.md` y `data/DATA_STATEMENT.md`. Se recibieron diez semillas por brazo: Dice medio de validación 0,7862 para UNet-AT y 0,8077 para la base. La evaluación final de test está pendiente.

## Detalles del modelo

- **Nombre:** UNet-AT (brazo `dbnet_at`); el brazo `dbnet` es su base de comparación.
- **Tipo:** red de segmentación de codificador-decodificador (U-Net compacta de tres niveles, anchos 16, 32 y 64, sin
  BatchNorm) con cabezas de probabilidad de núcleo y de umbral, y posproceso de expansión consistente (C2).
- **Tamaño:** 118.290 parámetros según el humo de entrenamiento; 17 más que la U-Net compacta del benchmark.
- **Entrenamiento:** desde cero, sin pesos preentrenados; Adam, tasa 1e-3, lote 2, hasta 30 épocas, paciencia 5.
- **Código:** `src/geometria_at.py`, `src/original_model.py`, `training/entrenar_original.py`; procedencia y
  SHA-256 en `experiments/procedencia.md`.
- **Licencia del código:** MIT (`LICENSE`); no cubre las fotografías ni las máscaras de HAZMAT-13.

## Uso previsto

Investigación y docencia: segmentar el rótulo de una mercancía peligrosa en una fotografía y comparar representaciones
de núcleo encogido frente a redes de segmentación estándar. Entrada: imagen RGB, ajustada a 256 × 256 con letterbox.
Salida: máscara binaria del rótulo.

## Usos no recomendados

- Decisiones de seguridad, de transporte o de respuesta a emergencias: el modelo no está validado para ellas.
- Operación portuaria o logística sin validación propia en imágenes del sitio: el test mide generalización interna.
- Imágenes fuera del dominio (sin rótulos o con fotografías muy distintas): el dashboard avisa con reglas de dominio
  cuyos umbrales no están calibrados.
- Lectura de la clase de riesgo: el modelo segmenta, no clasifica.

## Datos

HAZMAT-13 (versión «Original model»): 1.015 imágenes de entrenamiento, 218 de validación y 218 de prueba, particionadas
por grupos de similitud perceptual (semilla 20261106); 234 imágenes excluidas. Detalle, procedencia y licencia en
`data/DATA_STATEMENT.md`.

## Métricas

- **Primaria:** Dice por grupo en test (media por grupo y luego sobre 10 semillas).
- **Secundarias:** IoU y HD95 por grupo; Dice por tercil de área; brecha entre el tercil chico y el resto.
- **Cifras del original:** diez semillas por brazo verificadas en `data/avance_colab.json`. Mejor Dice de validación: UNet-AT 0,7862 ± 0,0192 y base 0,8077 ± 0,0109. Pareado por semilla, UNet-AT − base = −0,0215 (IC 95 % bootstrap de −0,036 a −0,010; Wilcoxon exacto p = 0,002; la base gana en las 10 semillas). Son resultados de validación con selección de época: en validación el mecanismo no mejora a su base. El test, que decide H1a y H1b, no se ha evaluado; la tabla de test de los rivales sigue siendo histórica.
- **Adecuación:** debe superar a la máscara vacía y a la mayoritaria (pendiente de medir).

## Limitaciones

- Un solo conjunto de datos; sin conjuntos adicionales ni cambio de distribución.
- La potencia recuperada corresponde a una referencia anterior: debe revisarse para la campaña actual. Diez semillas no garantizan el nivel exigido.
- Objetos pequeños: diferencias puntuales en las diez redes; los intervalos de la brecha entre tercil grande y pequeño excluyen cero en cinco redes.
- El teorema de inversión exacta vale para polígonos convexos con supuestos S1 y S2; en píxeles, la inversa es única
  solo hasta ±1 px.
- Calibración del original solo sobre el núcleo; energía solo estimada.

## Riesgos y sesgos

- Sesgos del conjunto: fotografías de origen y fecha desconocidos, sin metadatos de adquisición, con parte proveniente de
  competencias de robótica; geografía no documentada por imagen; clases desbalanceadas.
- Titularidad de las imágenes sin confirmar: no se redistribuyen.
- Riesgo de uso: una máscara incorrecta podría leerse como confirmación de ausencia de un rótulo.

## Reproducibilidad

Libro (`book/`), contenedor (`Dockerfile`), integración continua (`.github/workflows/e3.yml` en la raíz del repositorio) y lista de verificación en
el libro (capítulo de lista de verificación de reproducibilidad). Reproducción independiente de las cifras publicadas:
`experiments/reproduccion_independiente.py` (anexo A del libro).

## Integración local al 27 de noviembre

FCN y U-Net a0498 admiten inferencia ONNX; SegNet conserva un respaldo TorchScript verificado. El comparador de test utiliza predicciones precalculadas de nueve redes sobre 218 imágenes, con arquitectura, semilla e integridad registradas. El original se incorpora con la semilla 11 seleccionada por validación y posproceso C2 completo; VM-UNet sigue pendiente. Esta exploración implica exposición al test y no sustituye la evaluación final del original. La ficha accesible desde el tablero está en `assets/ficha_modelo.html`. Despliegue público: <https://hazmat13-e3.onrender.com>, con prueba de humo de 10 de 10 (anexo B del libro).
