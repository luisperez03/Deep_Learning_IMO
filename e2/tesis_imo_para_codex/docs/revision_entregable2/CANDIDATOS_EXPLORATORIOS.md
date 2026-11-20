# Candidatos exploratorios; selección pendiente

Fecha: 17 de noviembre de 2026. Fuentes primarias localizadas mediante búsqueda web dirigida por nombre; lectura de fichas y resúmenes, no evaluación completa. Esta lista no es el resultado del cribado sistemático ni el Top 10. No se asignan puntajes, cuartiles o conteos de citas sin comprobarlos.

| Candidato | Familia provisional | Mecanismo observado | Motivo para lectura completa y límite | Fuente primaria |
|---|---|---|---|---|
| DeepLabv3+ (2018) | CNN | Contexto multiescala y decodificación para bordes | Contraste con las CNN básicas; justificar excepción fundacional y fijar backbone | https://www.ecva.net/papers/eccv_2018/papers_ECCV/html/Liang-Chieh_Chen_Encoder-Decoder_with_Atrous_ECCV_2018_paper.php |
| UNet++ (2018) | CNN | Conexiones densas anidadas y supervisión profunda | Contrastar reconstrucción; aplicación médica, adaptación 2D por comprobar | https://arxiv.org/abs/1807.10165 |
| Attention U-Net (2018) | CNN con compuertas | Selección de regiones relevantes | Hipótesis sobre fondos complejos; no contar como familia Transformer; revisar dimensionalidad/configuración original | https://arxiv.org/abs/1804.03999 |
| SegFormer (2021) | Transformer | Codificador jerárquico y decodificador MLP | Configuración pequeña candidata; tamaño y costo deben medirse con variante fija | https://arxiv.org/abs/2105.15203 |
| TransUNet (2021; extensión 2024) | Híbrido CNN-Transformer | Contexto global y características locales | Distinguir versiones y configuraciones; dominio médico no demuestra eficacia HAZMAT | https://arxiv.org/abs/2102.04306 ; https://doi.org/10.1016/j.media.2024.103280 |
| VM-UNet (2024) | SSM visual | Bloques de estado visual y estructura U | Diversidad de mecanismo; verificar publicación, dependencias y presupuesto antes de incluir | https://arxiv.org/abs/2402.02491 |

No basta elegir estos seis y añadir otro nombre: faltan recuperación sistemática reciente, puntuación y viabilidad. Incluir alternativas encontradas aunque desplacen a los candidatos conocidos. FCN, SegNet y U-Net son referencias heredadas; las variantes compactas locales no equivalen a réplicas exactas de sus papers.

## Registro de búsqueda web dirigida de esta sesión

Buscador web general, no búsqueda ejecutada dentro de arXiv ni de Scopus. Consultas relevantes ejecutadas:

1. `site.arxiv.org SegFormer semantic segmentation 2105.15203`
2. `site.ecva.net DeepLabv3+ Encoder Decoder Atrous Separable Convolution semantic segmentation 2018`
3. `site.arxiv.org VM-UNet Vision Mamba UNet medical image segmentation`
4. `site.arxiv.org TransUNet Transformers Make Strong Encoders medical image segmentation`
5. `site.arxiv.org "UNet++" "Nested U-Net"`
6. `site.arxiv.org "Attention U-Net: Learning Where to Look"`
7. `site.openaccess.thecvf.com "Bilateral Guided Aggregation Network"` (sin fuente primaria utilizada en esta tabla).

Las búsquedas institucionales deben ampliar este conjunto; no hay conteo PRISMA derivado del buscador web.
