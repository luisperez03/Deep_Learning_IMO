# Atajos: contexto, rótulo borrado y color (filas 13 y 67)

Generado por `scripts/errores_atajos_e2.py atajos` con la salida de `colab/Atajos_Errores_E2.ipynb`. Validación, exploratorio, reglas fijadas antes (protocolo del 22/11/2026). Δ = Dice(condición) − Dice(limpia), IC95 por grupos; δ = 0,02; Holm sobre 20 pruebas.

| Red | Dice limpia | Sin contexto Δ [IC95] | Decisión | Gris Δ [IC95] | Decisión | Zona borrada predicha [IC95] | Falsa alarma en negativo [IC95] |
|---|---:|---:|---:|---:|---:|---:|---:|
| transunet_2d | 0,853 | -0,117 [-0,144; -0,091] | dependencia material | -0,332 [-0,382; -0,284] | dependencia material | 0,036 [0,022; 0,052] | 0,476 [0,413; 0,543] |
| fcn | 0,831 | 0,025 [0,012; 0,040] | sin dependencia material | -0,402 [-0,453; -0,352] | dependencia material | 0,014 [0,007; 0,023] | 0,321 [0,247; 0,395] |
| unet | 0,832 | 0,022 [0,009; 0,037] | sin dependencia material | -0,331 [-0,381; -0,284] | dependencia material | 0,014 [0,007; 0,024] | 0,307 [0,239; 0,378] |
| vmunet | 0,821 | -0,022 [-0,040; -0,004] | indeterminado | -0,288 [-0,325; -0,251] | dependencia material | 0,055 [0,033; 0,081] | 0,589 [0,529; 0,647] |
| segnet | 0,806 | 0,036 [0,021; 0,051] | sin dependencia material | -0,387 [-0,441; -0,334] | dependencia material | 0,015 [0,008; 0,025] | 0,330 [0,262; 0,402] |
| deeplabv3plus | 0,813 | 0,002 [-0,014; 0,019] | sin dependencia material | -0,368 [-0,418; -0,317] | dependencia material | 0,069 [0,045; 0,096] | 0,435 [0,366; 0,506] |
| unetplusplus | 0,810 | -0,000 [-0,019; 0,020] | sin dependencia material | -0,300 [-0,341; -0,258] | dependencia material | 0,082 [0,058; 0,109] | 0,481 [0,418; 0,543] |
| pidnet_s | 0,796 | 0,001 [-0,019; 0,023] | sin dependencia material | -0,324 [-0,367; -0,281] | dependencia material | 0,076 [0,053; 0,103] | 0,453 [0,385; 0,520] |
| segnext_t | 0,759 | -0,050 [-0,080; -0,020] | indeterminado | -0,313 [-0,359; -0,268] | dependencia material | 0,073 [0,049; 0,101] | 0,516 [0,441; 0,589] |
| segformer_b0 | 0,703 | -0,215 [-0,263; -0,168] | dependencia material | -0,419 [-0,469; -0,370] | dependencia material | 0,095 [0,065; 0,128] | 0,520 [0,446; 0,593] |

## Falsas alarmas en el negativo sintético, según la imagen limpia

Tasa de imágenes con al menos 50 píxeles predichos tras borrar los rótulos anotados, separando las que en la imagen limpia ya tenían exceso distante (lo que queda puede ser un rótulo sin anotar) de las que no.

| Red | Sin exceso distante previo (pares) | Con exceso distante previo (pares) | Píxeles predichos (mediana) |
|---|---:|---:|---:|
| transunet_2d | 0,370 (860) | 0,844 (230) | 32.0 |
| fcn | 0,107 (763) | 0,752 (327) | 0.0 |
| unet | 0,086 (722) | 0,704 (368) | 0.0 |
| vmunet | 0,458 (784) | 0,908 (306) | 138.0 |
| segnet | 0,103 (717) | 0,737 (373) | 0.0 |
| deeplabv3plus | 0,325 (898) | 0,906 (192) | 0.0 |
| unetplusplus | 0,388 (918) | 0,942 (172) | 4.5 |
| pidnet_s | 0,349 (929) | 0,950 (161) | 0.0 |
| segnext_t | 0,362 (818) | 0,941 (272) | 54.0 |
| segformer_b0 | 0,344 (678) | 0,801 (412) | 61.5 |

## Taxonomía exacta con componentes predichas (eje A)

Frecuencia [IC95 por grupos]; el exceso es distante si más de la mitad de sus píxeles está en componentes sin contacto con la máscara real dilatada 5 px.

| Red | Casos | Acuerdo con HD95 | omision total | deteccion desplazada | subsegmentacion | exceso adyacente | falso positivo distante | error mixto | Rótulos omitidos en imágenes con varios |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| transunet_2d | 188 | 0,862 | 0,000 [0,000; 0,000] | 0,011 [0,000; 0,036] | 0,325 [0,207; 0,453] | 0,330 [0,194; 0,472] | 0,197 [0,093; 0,320] | 0,138 [0,075; 0,207] | 32 |
| fcn | 246 | 0,785 | 0,008 [0,000; 0,022] | 0,008 [0,000; 0,021] | 0,240 [0,132; 0,363] | 0,248 [0,152; 0,351] | 0,285 [0,168; 0,407] | 0,211 [0,142; 0,288] | 1 |
| unet | 259 | 0,718 | 0,008 [0,000; 0,025] | 0,004 [0,000; 0,013] | 0,216 [0,121; 0,331] | 0,367 [0,242; 0,491] | 0,278 [0,166; 0,399] | 0,127 [0,077; 0,184] | 2 |
| vmunet | 253 | 0,751 | 0,004 [0,000; 0,013] | 0,016 [0,000; 0,042] | 0,269 [0,179; 0,373] | 0,387 [0,264; 0,512] | 0,182 [0,087; 0,286] | 0,142 [0,090; 0,197] | 30 |
| segnet | 324 | 0,741 | 0,015 [0,000; 0,041] | 0,009 [0,000; 0,025] | 0,176 [0,100; 0,266] | 0,386 [0,259; 0,515] | 0,296 [0,192; 0,412] | 0,117 [0,073; 0,169] | 2 |
| deeplabv3plus | 265 | 0,796 | 0,011 [0,000; 0,025] | 0,015 [0,000; 0,038] | 0,389 [0,272; 0,518] | 0,283 [0,159; 0,411] | 0,159 [0,072; 0,260] | 0,143 [0,093; 0,195] | 78 |
| unetplusplus | 279 | 0,771 | 0,036 [0,000; 0,087] | 0,014 [0,000; 0,032] | 0,373 [0,260; 0,503] | 0,326 [0,221; 0,435] | 0,107 [0,052; 0,172] | 0,143 [0,078; 0,212] | 119 |
| pidnet_s | 288 | 0,823 | 0,004 [0,000; 0,012] | 0,042 [0,018; 0,068] | 0,396 [0,293; 0,506] | 0,250 [0,154; 0,351] | 0,125 [0,062; 0,196] | 0,184 [0,128; 0,246] | 95 |
| segnext_t | 373 | 0,748 | 0,000 [0,000; 0,000] | 0,019 [0,002; 0,046] | 0,279 [0,180; 0,388] | 0,386 [0,263; 0,513] | 0,107 [0,044; 0,184] | 0,209 [0,132; 0,290] | 57 |
| segformer_b0 | 482 | 0,938 | 0,017 [0,002; 0,037] | 0,052 [0,015; 0,098] | 0,548 [0,448; 0,645] | 0,127 [0,061; 0,205] | 0,050 [0,018; 0,088] | 0,207 [0,137; 0,280] | 159 |

## Control: la imagen limpia reproduce el Dice de los diagnósticos

| Red | Pares | Máx. diferencia | Pares con diferencia > 0,001 |
|---|---:|---:|---:|
| transunet_2d | 1090 | 0,000 | 0 |
| fcn | 1090 | 0,000 | 0 |
| unet | 1090 | 0,000 | 0 |
| vmunet | 1090 | 0,000 | 0 |
| segnet | 1090 | 0,000 | 0 |
| deeplabv3plus | 1090 | 0,000 | 0 |
| unetplusplus | 1090 | 0,000 | 0 |
| pidnet_s | 1090 | 0,000 | 0 |
| segnext_t | 1090 | 0,128 | 792 |
| segformer_b0 | 1090 | 0,000 | 0 |
