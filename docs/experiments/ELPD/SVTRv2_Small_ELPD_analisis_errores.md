# Análisis de errores — SVTRv2-Small sobre ELPD/Kaggle

## Protocolo y trazabilidad

- **Conjunto:** ELPD/Kaggle, test estricto auditado (80 recortes: 76 GT verificados, 4 corregidos).
- **Modelo:** SVTRv2-Small, BGPD, `GTCDecoder(infer_gtc=True)`, `GTCLabelDecode`.
- **Modo:** zero-shot, sin fine-tuning; normalización `uppercase` y retener `A-Z0-9`.
- **Commit OpenOCR:** `1ccfc6ee6161f7133b192e16af3a9265273997ba`.
- **SHA-256 checkpoint:** `4c4e21c8eabd7a38519d9f10d6ffa286d470a07b1a8976109fe2b995a9414970`.
- **SHA-256 manifiesto:** `c22ea5efd511ca82a3889ed08c2b79ae44c655bbfbddaa844f12d800169cfc1c`.

## Métricas validadas

| Métrica | Resultado |
|---|---:|
| Imágenes | 80/80 |
| Aciertos exactos | 31 |
| Fallos exactos | 49 |
| EM | **38,75 %** |
| CER (micro) | **19,96 %** = 111/556 |
| GPU-forward por imagen (batch 8) | 3,305 ms |
| Parámetros | 12.518.159 |

## Tipología de errores (alineamiento mínimo de Levenshtein)

| Tipo de fallo | Imágenes |
|---|---:|
| Solo inserciones | 38 |
| Inserciones y sustituciones | 4 |
| Solo sustituciones | 3 |
| Solo eliminaciones | 3 |
| Eliminaciones y sustituciones | 1 |
| **Total** | **49** |

**Operaciones de edición:** 79 inserciones, 14 sustituciones y 18 eliminaciones (111 en total). Si existen alineamientos mínimos alternativos, el desglose por tipo de operación puede variar sin alterar la distancia ni el CER.

- 42 de 49 fallos tienen predicciones más largas que el GT.
- En 30 de 49 fallos, el GT completo aparece como subcadena de la predicción, indicando caracteres añadidos o fragmentos repetidos.
- Se observan errores con confianza alta; la puntuación de confianza no debe equipararse directamente a probabilidad calibrada de acierto.

## Casos prioritarios para la revisión visual

| Imagen | GT | Predicción | Distancia |
|---|---|---|---:|
| `test__EQ725QJ.jpg` | `EQ725QJ` | `GEO7250J250J250J250` | 14 |
| `test__CY726SV.jpg` | `CY726SV` | `CY726SV726SV726SV726` | 13 |
| `test__B961TAG.png` | `B961TAG` | `0` | 7 |
| `test__Z3010AZ.png` | `Z3010AZ` | `Z` | 6 |
| `test__B13GWF.png` | `B13GWF` | `B` | 5 |
| `test__EM727PD.jpg` | `EM727PD` | `NGEM727PDBD` | 4 |
| `test__I008646.png` | `IOO8646` | `01008646` | 4 |
| `test__CM101LW.png` | `CM101LW` | `8CH101LMG` | 4 |
| `test__CV194VA.jpg` | `CV194VA` | `DCV194VAZZ` | 3 |
| `test__EH577PH.jpg` | `EH577PH` | `WEH577PHBP` | 3 |

Los fallos graves incluyen repeticiones extensas (`EQ725QJ`, `CY726SV`) y truncamientos (`B961TAG`, `Z3010AZ`, `B13GWF`). Posibles factores como emblemas, ruido de fondo, perspectiva o comportamiento del decodificador **requieren inspección visual**.

## Comparación dentro de ELPD

| Métrica | PARSeq | SVTRv2-Tiny | SVTRv2-Small |
|---|---:|---:|---:|
| EM | 68,75 % | 28,75 % | 38,75 % |
| CER | 6,83 % | 21,04 % | 19,96 % |
| Aciertos | 55 | 23 | 31 |
| Parámetros | 23,83 M | 5,71 M | 12,52 M |
| GPU-forward medio/imagen | 4,394 ms | 3,288 ms | 3,305 ms |

**Tiny vs Small por imagen:** 14 ambas correctas, 17 solo Small correcta, 9 solo Tiny correcta, 40 ambas erróneas. **PARSeq vs Small:** 28 ambas correctas, 3 solo Small correcta, 27 solo PARSeq correcta, 22 ambas erróneas.

Los tiempos GPU-forward registrados no constituyen aún una comparación de latencia extremo a extremo, ya que no incluyen todo el postprocesado ni un benchmark común con calentamiento.

## Correcciones de GT

De las cuatro etiquetas corregidas, Small acierta una (`YBZ5711`) y falla tres. El subconjunto es demasiado pequeño para generalizaciones.

## Estado

**Análisis cuantitativo verificado. Auditoría visual pendiente.** Utilizar `scripts/baselines/analyze_elpd_baseline_errors.py` para generar hojas de contacto con las 49 predicciones incorrectas y analizar sus imágenes originales.

## Listado completo de las 49 predicciones incorrectas

| Imagen | GT | Predicción normalizada | D | Tipo |
|---|---|---|---:|---|
| `test__EQ725QJ.jpg` | `EQ725QJ` | `GEO7250J250J250J250` | 14 | insertion+substitution |
| `test__CY726SV.jpg` | `CY726SV` | `CY726SV726SV726SV726` | 13 | insertion |
| `test__B961TAG.png` | `B961TAG` | `0` | 7 | deletion+substitution |
| `test__Z3010AZ.png` | `Z3010AZ` | `Z` | 6 | deletion |
| `test__B13GWF.png` | `B13GWF` | `B` | 5 | deletion |
| `test__CM101LW.png` | `CM101LW` | `8CH101LMG` | 4 | insertion+substitution |
| `test__EM727PD.jpg` | `EM727PD` | `NGEM727PDBD` | 4 | insertion |
| `test__I008646.png` | `IOO8646` | `01008646` | 4 | insertion+substitution |
| `test__CV194VA.jpg` | `CV194VA` | `DCV194VAZZ` | 3 | insertion |
| `test__EH577PH.jpg` | `EH577PH` | `WEH577PHBP` | 3 | insertion |
| `test__5888881.png` | `5888881` | `1588881` | 2 | substitution |
| `test__AB323VJ.png` | `AB323VJ` | `8AB323VJ2` | 2 | insertion |
| `test__ANAL334.png` | `ANAL334` | `BANSAL334` | 2 | insertion |
| `test__ES484LJ.jpg` | `ES484LJ` | `HES484LJB` | 2 | insertion |
| `test__FF788NK.jpg` | `FF788NK` | `8FF788NK8` | 2 | insertion |
| `test__IICE3083.png` | `11CE3083` | `JICE3083` | 2 | substitution |
| `test__MWWK908.png` | `MWWK908` | `BMW3WK908` | 2 | insertion |
| `test__WW921MX.jpg` | `WW921MX` | `WWW921HX` | 2 | insertion+substitution |
| `train__VS236891.png` | `VS236891` | `DVS2368910` | 2 | insertion |
| `test__34EN9199.png` | `34EN9199` | `934EN9199` | 1 | insertion |
| `test__5B40001.png` | `5B40001` | `45B40001` | 1 | insertion |
| `test__75N1960G.png` | `75N1960G` | `875N1960G` | 1 | insertion |
| `test__8427XX29.jpg` | `8427XX29` | `88427XX29` | 1 | insertion |
| `test__AE330NJ.jpg` | `AE330NJ` | `AE330NJ1` | 1 | insertion |
| `test__AX128FT.png` | `AX128FT` | `8AX128FT` | 1 | insertion |
| `test__BA999ZZ.png` | `BA999ZZ` | `BAO999ZZ` | 1 | insertion |
| `test__BL654BP.png` | `BL654BP` | `BLO654BP` | 1 | insertion |
| `test__BM602BT.png` | `BM602BT` | `BMM602BT` | 1 | insertion |
| `test__CK0511.png` | `CK0511` | `MCK0511` | 1 | insertion |
| `test__CN498JC.png` | `CN498JC` | `8CN498JC` | 1 | insertion |
| `test__CX408PX.jpg` | `CX408PX` | `CCX408PX` | 1 | insertion |
| `test__DANKE82.png` | `DANKE82` | `DANSKE82` | 1 | insertion |
| `test__EJQ588.png` | `EJQ588` | `6EJQ588` | 1 | insertion |
| `test__ESCC690.png` | `ESCC690` | `LESCC690` | 1 | insertion |
| `test__FR697HF.jpg` | `FR697HF` | `FR697HFH` | 1 | insertion |
| `test__FS633TK.jpg` | `FS633TK` | `6FS633TK` | 1 | insertion |
| `test__GB1ECB.png` | `GB1ECB` | `GBO1ECB` | 1 | insertion |
| `test__H864JGM.png` | `H864JGM` | `CH864JGM` | 1 | insertion |
| `test__KYM3141.png` | `KYM3141` | `CKYM3141` | 1 | insertion |
| `test__LBHT71.png` | `LBHT71` | `1LBHT71` | 1 | insertion |
| `test__MH49XK.png` | `MH49XK` | `LMH49XK` | 1 | insertion |
| `test__PI36KOG.png` | `P136KOG` | `PI36KOG` | 1 | substitution |
| `test__SLF9995.png` | `SLF9995` | `SSLF9995` | 1 | insertion |
| `test__VI496AN.png` | `VI496AN` | `VIE496AN` | 1 | insertion |
| `test__WANNE685.JPG` | `WANNE685` | `WANE685` | 1 | deletion |
| `test__WW697BH.jpg` | `WW697BH` | `WWW697BH` | 1 | insertion |
| `train__381ATK83.png` | `381ATK83` | `8381ATK83` | 1 | insertion |
| `train__75N1960G_1.png` | `75N1960G` | `875N1960G` | 1 | insertion |
| `train__BL33635_1.png` | `BL33635` | `OBL33635` | 1 | insertion |
