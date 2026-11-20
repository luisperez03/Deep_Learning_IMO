# Limitación de SegFormer-B0 por área del rótulo (datos reales de validación)

Veredicto a priori: **no_confirmada**. Protocolo SHA-256 `9f07a9d3663b`.
Cortes de terciles (fracción de área del grupo): 0.0195, 0.0411; grupos por tercil [41, 40, 41].

| Red | Dice T1 | Dice T3 | Δ T3-T1 [IC95] | DE semillas Δ |
|---|---|---|---|---|
| fcn | 0.779 | 0.866 | 0.086 [0.011; 0.153] | 0.022 |
| segnet | 0.747 | 0.844 | 0.097 [0.017; 0.170] | 0.017 |
| unet | 0.782 | 0.864 | 0.081 [0.004; 0.150] | 0.016 |
| deeplabv3plus | 0.772 | 0.845 | 0.073 [-0.007; 0.148] | 0.022 |
| unetplusplus | 0.776 | 0.835 | 0.059 [-0.025; 0.138] | 0.026 |
| pidnet_s | 0.760 | 0.813 | 0.053 [-0.037; 0.136] | 0.026 |
| segnext_t | 0.699 | 0.819 | 0.120 [0.030; 0.207] | 0.009 |
| segformer_b0 | 0.661 | 0.743 | 0.082 [-0.030; 0.194] | 0.028 |
| transunet_2d | 0.814 | 0.871 | 0.057 [-0.014; 0.119] | 0.016 |
| vmunet | 0.750 | 0.852 | 0.102 [0.018; 0.182] | 0.026 |

| Comparador | C = Δ guía - Δ c [IC95] | p Holm | Nivel T1 guía - c [IC95] |
|---|---|---|---|
| unet | 0.001 [-0.076; 0.082] | 1.0000 | -0.121 [-0.184; -0.064] |
| deeplabv3plus | 0.010 [-0.057; 0.076] | 1.0000 | -0.111 [-0.165; -0.063] |
| segnext_t | -0.038 [-0.095; 0.018] | 0.5484 | -0.038 [-0.080; 0.001] |
