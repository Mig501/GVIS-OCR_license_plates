# SVTRv2-Tiny — análisis de errores en ELPD/Kaggle

## Identificación del experimento

- Dataset: ELPD/Kaggle, test estricto auditado (80 imágenes; 76 GT verificados, 4 corregidos).
- Modelo: SVTRv2-Tiny (BGPD, GTCDecoder, infer_gtc=True, GTCLabelDecode); sin fine-tuning.
- Commit OpenOCR: `1ccfc6ee6161f7133b192e16af3a9265273997ba`.
- SHA-256 checkpoint: `684f23bc3a2b1f724cadfd275e1a185456dd51b5e56c8830780d1998e90f77f6`.
- SHA-256 manifiesto: `c22ea5efd511ca82a3889ed08c2b79ae44c655bbfbddaa844f12d800169cfc1c`.
- Normalización: `uppercase and retain ASCII A-Z0-9`.

## Resultados

| Indicador | Valor |
|---|---:|
| Predicciones | 80/80 |
| EM | 23/80 = 28,75 % |
| CER micro | 117/556 = 21,04 % |
| Predicciones fallidas | 57 |
| Latencia GPU-forward, batch 8 | 3,288 ms/imagen (sin postprocesado) |
| Parámetros | 5.710.751 |

## Descomposición de las 117 ediciones (Levenshtein)

| Operación | Cantidad |
|---|---:|
| Inserciones | 90 |
| Sustituciones | 16 |
| Eliminaciones | 11 |

## Clases de fallos por imagen

| Clase | Imágenes |
|---|---:|
| Solo inserciones | 42 |
| Solo sustituciones | 1 |
| Solo eliminaciones | 3 |
| Combinado (insercion + sustitucion) | 11 |

- 28/57 predicciones incorrectas contienen exactamente el GT completo con uno o más caracteres adicionales **solo al principio**.
- 53/57 predicciones incorrectas son más largas que el GT; 3 son más cortas y 1 conserva la longitud.
- La tendencia a añadir caracteres podría relacionarse con bandas, símbolos o fondo; esta hipótesis necesita inspección visual de los recortes y no demuestra la causa.
- 2/4 etiquetas corregidas manualmente se reconocieron por completo.

## Casos prioritarios para revisión visual

| Imagen | GT | Predicción | Ediciones | Confianza |
|---|---|---|---:|---:|
| `test__EQ725QJ.jpg` | `EQ725QJ` | `E0725QJ250J250J25` | 11 | 0.671 |
| `test__FS633TK.jpg` | `FS633TK` | `LFS633TK33TK33TK3` | 10 | 0.683 |
| `test__BM602BT.png` | `BM602BT` | `BM` | 5 | 0.697 |
| `test__KA761ET.png` | `KA761ET` | `KA` | 5 | 0.946 |
| `test__I008646.png` | `IOO8646` | `11008646` | 4 | 0.830 |
| `test__AL193VP.jpg` | `AL193VP` | `1AL193WPER` | 4 | 0.644 |
| `test__PI36KOG.png` | `P136KOG` | `PPI36KOG` | 2 | 0.750 |

## Recomendaciones

1. Revisar visualmente las 57 imágenes erróneas, especialmente los prefijos añadidos, las repeticiones anómalas y los truncamientos.
2. Contrastar cada caso con PARSeq y luego SVTRv2-Small y MDiff4STR-Small, siempre sobre las mismas 80 imágenes.
3. Conservar las predicciones originales y el EM/CER sin correcciones; los ensayos de normalización adicional deben documentarse como ablaciones separadas.
4. Medir tiempos con calentamiento y protocolo idéntico (GPU-forward vs extremo a extremo) antes de publicar rankings de velocidad.

## Detalle de las 57 predicciones incorrectas

| Imagen | GT | Predicción | Ediciones | Clase |
|---|---|---|---:|---|
| `test__2348XR25.jpg` | `2348XR25` | `12348XR25` | 1 | Solo inserciones |
| `test__34EN9199.png` | `34EN9199` | `834EN9199` | 1 | Solo inserciones |
| `test__381ATK83_1.png` | `381ATK83` | `1381ATK83` | 1 | Solo inserciones |
| `test__55SG53.jpg` | `55SG53` | `1555653` | 3 | Combinado (insercion + sustitucion) |
| `test__5658YA22.jpg` | `5658YA22` | `45658YA22` | 1 | Solo inserciones |
| `test__5888881.png` | `5888881` | `15808881` | 2 | Combinado (insercion + sustitucion) |
| `test__5B40001.png` | `5B40001` | `15B40001` | 1 | Solo inserciones |
| `test__75N1960G.png` | `75N1960G` | `875N1960G` | 1 | Solo inserciones |
| `train__75N1960G_1.png` | `75N1960G` | `875N1960G` | 1 | Solo inserciones |
| `test__8427XX29.jpg` | `8427XX29` | `18427XX29` | 1 | Solo inserciones |
| `test__AB323VJ.png` | `AB323VJ` | `BAB323VJE` | 2 | Solo inserciones |
| `test__AC508V.png` | `AC508V` | `AC508VE` | 1 | Solo inserciones |
| `test__AE330NJ.jpg` | `AE330NJ` | `MAE330NJ` | 1 | Solo inserciones |
| `test__AL193VP.jpg` | `AL193VP` | `1AL193WPER` | 4 | Combinado (insercion + sustitucion) |
| `test__ANAL334.png` | `ANAL334` | `BANAL334` | 1 | Solo inserciones |
| `test__AX128FT.png` | `AX128FT` | `8AX128FT` | 1 | Solo inserciones |
| `test__B13GWF.png` | `B13GWF` | `BM13GWF` | 1 | Solo inserciones |
| `test__B21GSB.png` | `B21GSB` | `B216SB` | 1 | Solo sustituciones |
| `test__B225RTK.png` | `B225RTK` | `118225RTK` | 3 | Combinado (insercion + sustitucion) |
| `test__B961TAG.png` | `B961TAG` | `10B961TAG` | 2 | Solo inserciones |
| `test__B96SXV_1.png` | `B96SXV` | `LB96SXV` | 1 | Solo inserciones |
| `train__B96SXV.png` | `B96SXV` | `EB96SXV` | 1 | Solo inserciones |
| `train__BL33635_1.png` | `BL33635` | `COBL336350` | 3 | Solo inserciones |
| `test__BM602BT.png` | `BM602BT` | `BM` | 5 | Solo eliminaciones |
| `test__BZ310SW.jpg` | `BZ310SW` | `HBZ310SW` | 1 | Solo inserciones |
| `test__CK0511.png` | `CK0511` | `MCK0511` | 1 | Solo inserciones |
| `test__CM101LW.png` | `CM101LW` | `18CH101LW` | 3 | Combinado (insercion + sustitucion) |
| `test__CN498JC.png` | `CN498JC` | `BCN498JCB` | 2 | Solo inserciones |
| `test__CV194VA.jpg` | `CV194VA` | `1CV194VAR` | 2 | Solo inserciones |
| `test__CX408PX.jpg` | `CX408PX` | `ACX408PX` | 1 | Solo inserciones |
| `test__CY726SV.jpg` | `CY726SV` | `CY726V` | 1 | Solo eliminaciones |
| `test__EH577PH.jpg` | `EH577PH` | `WEH577PH` | 1 | Solo inserciones |
| `test__EJQ588.png` | `EJQ588` | `BEJQ588` | 1 | Solo inserciones |
| `test__EM727PD.jpg` | `EM727PD` | `JEM727PDT` | 2 | Solo inserciones |
| `test__EQ725QJ.jpg` | `EQ725QJ` | `E0725QJ250J250J25` | 11 | Combinado (insercion + sustitucion) |
| `test__ES484LJ.jpg` | `ES484LJ` | `HES484LJ9` | 2 | Solo inserciones |
| `test__ESCC690.png` | `ESCC690` | `LESCC690` | 1 | Solo inserciones |
| `test__FC882MC.jpg` | `FC882MC` | `LFC882MC2` | 2 | Solo inserciones |
| `test__FF788NK.jpg` | `FF788NK` | `FFF788NK8` | 2 | Solo inserciones |
| `test__FL986ZT.jpg` | `FL986ZT` | `FL9862T5` | 2 | Combinado (insercion + sustitucion) |
| `test__FP786YF.jpg` | `FP786YF` | `FFP786YFLE` | 3 | Solo inserciones |
| `test__FR697HF.jpg` | `FR697HF` | `44697HFI` | 3 | Combinado (insercion + sustitucion) |
| `test__FS633TK.jpg` | `FS633TK` | `LFS633TK33TK33TK3` | 10 | Solo inserciones |
| `test__I008646.png` | `IOO8646` | `11008646` | 4 | Combinado (insercion + sustitucion) |
| `test__KA761ET.png` | `KA761ET` | `KA` | 5 | Solo eliminaciones |
| `test__KYM3141.png` | `KYM3141` | `LKYM3141` | 1 | Solo inserciones |
| `test__LBHT71.png` | `LBHT71` | `IBLBHT71` | 2 | Solo inserciones |
| `test__MH49XK.png` | `MH49XK` | `LMH49XK` | 1 | Solo inserciones |
| `test__MWWK908.png` | `MWWK908` | `BMWWK908` | 1 | Solo inserciones |
| `test__N1966.png` | `N1966` | `18N1966` | 2 | Solo inserciones |
| `test__PI36KOG.png` | `P136KOG` | `PPI36KOG` | 2 | Combinado (insercion + sustitucion) |
| `train__TR95XMI.png` | `TR95XMI` | `ATR95XMI` | 1 | Solo inserciones |
| `test__VI496AN.png` | `VI496AN` | `VIE496AN` | 1 | Solo inserciones |
| `train__VS236891.png` | `VS236891` | `DVS2368910` | 2 | Solo inserciones |
| `test__WW697BH.jpg` | `WW697BH` | `WWW697BH` | 1 | Solo inserciones |
| `test__WW921MX.jpg` | `WW921MX` | `1WW921HX` | 2 | Combinado (insercion + sustitucion) |
| `test__Z3010AZ.png` | `Z3010AZ` | `1Z3010AZ` | 1 | Solo inserciones |
