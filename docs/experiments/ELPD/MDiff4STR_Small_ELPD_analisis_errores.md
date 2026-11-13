# Análisis de errores — MDiff4STR-Small sobre ELPD/Kaggle

## Protocolo y trazabilidad

- **Conjunto:** ELPD/Kaggle, test estricto auditado (80 recortes: 76 etiquetas verificadas y 4 corregidas).
- **Modelo:** MDiff4STR-Small; `MDiffDecoder`, 3 pasos de difusión, `sample_k=3`, modo semi-autoregresivo.
- **Modo:** zero-shot, sin fine-tuning; `seed=42`, `batch_size=8` y normalización a mayúsculas conservando únicamente `A-Z0-9`.
- **Commit OpenOCR:** `1ccfc6ee6161f7133b192e16af3a9265273997ba`.
- **SHA-256 checkpoint:** `ebc96f11e5739eae98491f7c8c140157ed12b35dafa220511ef9d399cc20a937`.
- **SHA-256 configuración:** `0abdd478db64fa06b0574f56f8c8620800bab511af0af94861b9ea73bcb219bf`.
- **SHA-256 manifiesto:** `c22ea5efd511ca82a3889ed08c2b79ae44c655bbfbddaa844f12d800169cfc1c`.
- **Entorno:** PyTorch 2.14.1+cu126, Python 3.14.3, NVIDIA A100-PCIE-40GB.

## Métricas verificadas

| Métrica | Resultado |
|---|---:|
| Imágenes evaluadas | 80/80 |
| Matrículas correctas | 37 |
| Matrículas incorrectas | 43 |
| Exact Match (EM) | **46,25 %** |
| CER micro | **14,03 %** = 78/556 |
| GPU-forward por imagen (batch 8) | 2,531 ms |
| Parámetros | 18.891.072 |

Las métricas EM y CER se han recalculado a partir de `predictions.csv` y coinciden con el `summary.json`. La latencia mide el *forward* GPU de tres pasos de difusión, pero **no incluye el preprocesado ni el posprocesado**; no es una comparación de extremo a extremo.

## Tipología de errores (alineamiento mínimo de Levenshtein)

| Tipo de fallo | Matrículas |
|---|---:|
| Solo inserción | 15 |
| inserción + eliminación | 11 |
| inserción + sustitución | 8 |
| Solo eliminación | 5 |
| Solo sustitución | 2 |
| inserción + sustitución + eliminación | 1 |
| sustitución + eliminación | 1 |
| **Total** | **43** |

**Operaciones acumuladas:** 44 inserciones, 16 sustituciones y 18 eliminaciones (**78** en total).

- Predicciones incorrectas de mayor longitud: **27**; igual longitud: **10**; menor longitud: **6**.
- La cadena GT completa aparece dentro de la predicción en **12** de los 43 errores.
- El desglose por tipo depende del alineamiento mínimo escogido en secuencias repetidas: puede haber varias alineaciones óptimas equivalentes con igual CER.

## Casos prioritarios para inspección visual

| Imagen | GT | Predicción normalizada | Distancia |
|---|---|---|---:|
| `test__55SG53.jpg` | `55SG53` | `555665` | 4 |
| `test__EM727PD.jpg` | `EM727PD` | `HEM777PDET` | 4 |
| `test__FR697HF.jpg` | `FR697HF` | `FFR669HFF` | 4 |
| `test__AB323VJ.png` | `AB323VJ` | `BAB33VJS` | 3 |
| `test__ANAL334.png` | `ANAL334` | `BANSA334` | 3 |
| `test__AX128FT.png` | `AX128FT` | `BAX12FTB` | 3 |
| `test__I008646.png` | `IOO8646` | `1008646` | 3 |
| `train__VS236891.png` | `VS236891` | `UVS236910` | 3 |
| `test__2348XR25.jpg` | `2348XR25` | `0248XR25` | 2 |
| `test__34EN9199.png` | `34EN9199` | `834E9199` | 2 |
| `test__5888881.png` | `5888881` | `15888811` | 2 |
| `test__75N1960G.png` | `75N1960G` | `775N196G` | 2 |

Los casos prioritarios comprenden caracteres añadidos, confusiones y omisiones. Su causa física (escudos, textura, falta de nitidez, perspectiva o decodificador) **no puede establecerse únicamente con el CSV**; requiere revisión de las imágenes originales.

## Comparación de baselines sobre las mismas 80 imágenes

| Modelo | EM ↑ | CER ↓ | Aciertos | Parámetros | GPU-forward ms/imagen* |
|---|---:|---:|---:|---:|---:|
| PARSeq | 68.75 % | 6.83 % | 55/80 | 23,832,671 | 4.394 |
| SVTRv2-Tiny | 28.75 % | 21.04 % | 23/80 | 5,710,751 | 3.288 |
| SVTRv2-Small | 38.75 % | 19.96 % | 31/80 | 12,518,159 | 3.305 |
| MDiff4STR-Small | 46.25 % | 14.03 % | 37/80 | 18,891,072 | 2.531 |

*Latencias provisionales, no medidas con un benchmark extremo a extremo homogéneo.*

| Contraste | Ambos correctos | Solo MDiff4STR correcto | Solo el otro correcto | Ambos incorrectos |
|---|---:|---:|---:|---:|
| Frente a PARSeq | 33 | 4 | 22 | 21 |
| Frente a SVTRv2-Tiny | 17 | 20 | 6 | 37 |
| Frente a SVTRv2-Small | 23 | 14 | 8 | 35 |

- **Fallos comunes a los cuatro modelos:** 15 imágenes.
- **Aciertos comunes a los cuatro modelos:** 12 imágenes.
- **MDiff4STR acierta y PARSeq falla:** `test__5B40001.png`, `test__BZ310SW.jpg`, `test__CY726SV.jpg`, `test__GB1ECB.png`.

## Subconjunto de etiquetas corregidas

De las **4 etiquetas corregidas**, MDiff4STR-Small acierta **1** y falla **3**. Este subconjunto es demasiado pequeño para inferencias generales.

## Estado y próximos pasos

**Análisis cuantitativo validado; inspección visual pendiente.** Generar las hojas de contacto con `scripts/baselines/analyze_elpd_baseline_errors.py` y revisar prioritariamente las 43 imágenes fallidas. Mantener las predicciones sin reglas correctoras para preservar el baseline zero-shot.

## Listado completo de las 43 predicciones incorrectas

| Imagen | GT | Predicción normalizada | D | Tipo |
|---|---|---|---:|---|
| `test__55SG53.jpg` | `55SG53` | `555665` | 4 | inserción + sustitución + eliminación |
| `test__EM727PD.jpg` | `EM727PD` | `HEM777PDET` | 4 | inserción + sustitución |
| `test__FR697HF.jpg` | `FR697HF` | `FFR669HFF` | 4 | inserción + sustitución |
| `test__AB323VJ.png` | `AB323VJ` | `BAB33VJS` | 3 | inserción + eliminación |
| `test__ANAL334.png` | `ANAL334` | `BANSA334` | 3 | inserción + eliminación |
| `test__AX128FT.png` | `AX128FT` | `BAX12FTB` | 3 | inserción + eliminación |
| `test__I008646.png` | `IOO8646` | `1008646` | 3 | Solo sustitución |
| `train__VS236891.png` | `VS236891` | `UVS236910` | 3 | inserción + eliminación |
| `test__2348XR25.jpg` | `2348XR25` | `0248XR25` | 2 | inserción + eliminación |
| `test__34EN9199.png` | `34EN9199` | `834E9199` | 2 | inserción + eliminación |
| `test__5888881.png` | `5888881` | `15888811` | 2 | inserción + sustitución |
| `test__75N1960G.png` | `75N1960G` | `775N196G` | 2 | inserción + eliminación |
| `test__B13GWF.png` | `B13GWF` | `BM13GFF` | 2 | inserción + sustitución |
| `test__B225RTK.png` | `B225RTK` | `B25RTKK` | 2 | inserción + eliminación |
| `test__BL654BP.png` | `BL654BP` | `BL0654P` | 2 | inserción + eliminación |
| `test__CM101LW.png` | `CM101LW` | `CH101LW3` | 2 | inserción + sustitución |
| `test__CN498JC.png` | `CN498JC` | `BCN498JCH` | 2 | Solo inserción |
| `test__CV194VA.jpg` | `CV194VA` | `BCV194VAZ` | 2 | Solo inserción |
| `test__FP786YF.jpg` | `FP786YF` | `FF786YFB` | 2 | inserción + sustitución |
| `test__IICE3083.png` | `11CE3083` | `ICE3083` | 2 | sustitución + eliminación |
| `test__PI36KOG.png` | `P136KOG` | `PPI36KOG` | 2 | inserción + sustitución |
| `test__VI496AN.png` | `VI496AN` | `VI549AN` | 2 | inserción + eliminación |
| `test__WW921MX.jpg` | `WW921MX` | `WWW921HX` | 2 | inserción + sustitución |
| `train__TR95XMI.png` | `TR95XMI` | `TTR95MI` | 2 | inserción + eliminación |
| `test__AE330NJ.jpg` | `AE330NJ` | `AE330NJM` | 1 | Solo inserción |
| `test__CK0511.png` | `CK0511` | `MCK0511` | 1 | Solo inserción |
| `test__CX408PX.jpg` | `CX408PX` | `ICX408PX` | 1 | Solo inserción |
| `test__DANKE82.png` | `DANKE82` | `DANSKE82` | 1 | Solo inserción |
| `test__EH577PH.jpg` | `EH577PH` | `EH57PH` | 1 | Solo eliminación |
| `test__ES484LJ.jpg` | `ES484LJ` | `ES444LJ` | 1 | Solo sustitución |
| `test__ESCC690.png` | `ESCC690` | `BESCC690` | 1 | Solo inserción |
| `test__FF788NK.jpg` | `FF788NK` | `FFF788NK` | 1 | Solo inserción |
| `test__FS633TK.jpg` | `FS633TK` | `FFS633TK` | 1 | Solo inserción |
| `test__LBHT71.png` | `LBHT71` | `LBBHT71` | 1 | Solo inserción |
| `test__MH49XK.png` | `MH49XK` | `MH49K` | 1 | Solo eliminación |
| `test__MWWK908.png` | `MWWK908` | `BMWWK908` | 1 | Solo inserción |
| `test__N1966.png` | `N1966` | `BN1966` | 1 | Solo inserción |
| `test__TR95XMI_1.png` | `TR95XMI` | `TR95XM` | 1 | Solo eliminación |
| `test__WANNE685.JPG` | `WANNE685` | `WANNE6885` | 1 | Solo inserción |
| `test__WW697BH.jpg` | `WW697BH` | `WWW697BH` | 1 | Solo inserción |
| `test__Z3010AZ.png` | `Z3010AZ` | `ZZ3010AZ` | 1 | Solo inserción |
| `train__BL33635_1.png` | `BL33635` | `BL3635` | 1 | Solo eliminación |
| `train__TR95XMI_2.png` | `TR95XMI` | `TR95XM` | 1 | Solo eliminación |
