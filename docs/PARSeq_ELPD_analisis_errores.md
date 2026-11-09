# PARSeq — Análisis inicial de errores en ELPD/Kaggle

**Dataset:** subconjunto de 80 matrículas con GT auditado (76 verified, 4 corrected).  
**Modelo:** PARSeq preentrenado, inferencia zero-shot, sin fine-tuning.  
**Checkpoint SHA-256:** `bb5792a68e367476abca029cbf8699abc805f3d3dc7e57aae45c8ec4f7b7cd00`.  
**Manifest SHA-256:** `c22ea5efd511ca82a3889ed08c2b79ae44c655bbfbddaa844f12d800169cfc1c`.

## Métricas globales

- **EM:** 55/80 = **68,75 %**; errores: **25**.
- **CER:** 38/556 = **6,83 %**.
- De las 4 anotaciones corregidas, **1** se reconoció correctamente.

## Taxonomía del error (por imagen)

| Clasificación | Imágenes |
|---|---:|
| Solo inserciones | 16 |
| Solo sustituciones | 3 |
| Solo eliminaciones | 2 |
| Combinaciones | 4 |
| **Total** | **25** |

En una alineación óptima de Levenshtein se contabilizan **23 inserciones**, **9 sustituciones** y **6 eliminaciones**, sumando 38 operaciones. 
**20 de 25** predicciones incorrectas incluyen al menos una inserción.

## Las 25 matrículas incorrectas

| Imagen | GT | Predicción OCR | Distancia | Tipo | Confianza |
|---|---|---|---:|---|---:|
| `test__GB1ECB.png` | `GB1ECB` | `GB` | 4 | deletion | 0.361 |
| `test__IICE3083.png` | `11CE3083` | `EIICE30837` | 4 | mixed | 0.087 |
| `test__I008646.png` | `IOO8646` | `1008646` | 3 | substitution | 0.845 |
| `test__ANAL334.png` | `ANAL334` | `BANSAL334` | 2 | insertion | 0.132 |
| `test__CM101LW.png` | `CM101LW` | `BCH101LW` | 2 | mixed | 0.290 |
| `test__FR697HF.jpg` | `FR697HF` | `FR697HFHF` | 2 | insertion | 0.200 |
| `test__TR95XMI_1.png` | `TR95XMI` | `LTR95XM` | 2 | mixed | 0.195 |
| `test__WW921MX.jpg` | `WW921MX` | `WW921HX2` | 2 | mixed | 0.154 |
| `test__5888881.png` | `5888881` | `58888881` | 1 | insertion | 0.659 |
| `test__5B40001.png` | `5B40001` | `5840001` | 1 | substitution | 0.562 |
| `test__BL654BP.png` | `BL654BP` | `BL0654BP` | 1 | insertion | 0.524 |
| `test__BZ310SW.jpg` | `BZ310SW` | `BZ310SWH` | 1 | insertion | 0.530 |
| `test__CN498JC.png` | `CN498JC` | `CN498JCB` | 1 | insertion | 0.625 |
| `test__CV194VA.jpg` | `CV194VA` | `CV194VA2` | 1 | insertion | 0.371 |
| `test__CY726SV.jpg` | `CY726SV` | `CY726SVM` | 1 | insertion | 0.382 |
| `test__DANKE82.png` | `DANKE82` | `DANSKE82` | 1 | insertion | 0.703 |
| `test__ES484LJ.jpg` | `ES484LJ` | `BES484LJ` | 1 | insertion | 0.395 |
| `test__ESCC690.png` | `ESCC690` | `LESCC690` | 1 | insertion | 0.438 |
| `test__MWWK908.png` | `MWWK908` | `MWWWK908` | 1 | insertion | 0.270 |
| `test__PI36KOG.png` | `P136KOG` | `PI36KOG` | 1 | substitution | 0.710 |
| `test__WANNE685.JPG` | `WANNE685` | `WANSNE685` | 1 | insertion | 0.185 |
| `test__WW697BH.jpg` | `WW697BH` | `WWW697BH` | 1 | insertion | 0.257 |
| `test__Z3010AZ.png` | `Z3010AZ` | `ZZ3010AZ` | 1 | insertion | 0.305 |
| `train__TR95XMI_2.png` | `TR95XMI` | `TR95XM` | 1 | deletion | 0.729 |
| `train__VS236891.png` | `VS236891` | `VS2368910` | 1 | insertion | 0.188 |

## Casos que conviene revisar visualmente primero

- **`test__IICE3083.png`**: cuatro operaciones; GT corregido y posible lectura de información exterior a la matrícula. *Causa aún no verificada visualmente.*
- **`test__GB1ECB.png`**: reconoce solo `GB`, con cuatro eliminaciones; revisar resolución, orientación, iluminación y segmentación.
- **`test__I008646.png`**: tres sustituciones al confundir `I` y `O` con `1` y `0`; el GT fue corregido manualmente.
- **`test__FR697HF.jpg`**: duplica `HF`; revisar si hay texto en el borde o redundancia de caracteres.
- **`test__WW921MX.jpg`**: inserción y sustitución; revisar caracteres finales.
- **`test__BZ310SW.jpg`**, **`test__CN498JC.png`** y **`test__CV194VA.jpg`**: inserciones al final de la secuencia.

## Criterios para auditoría visual

Al ver las imágenes, clasificar cada caso por posibles factores visuales: baja resolución, desenfoque, reflejos, iluminación deficiente, perspectiva, fondo/texto adicional, caracteres similares, recorte parcial o GT cuestionable. Estas causas **no pueden deducirse solo del CSV de predicciones**.

**Precauciones:** La clasificación de operaciones usa una alineación óptima determinista; con caracteres repetidos puede haber otras alineaciones igual de válidas. La confianza del modelo no equivale a una probabilidad calibrada de acierto. No alterar etiquetas, imágenes ni predicciones sin una nueva auditoría explícita.
