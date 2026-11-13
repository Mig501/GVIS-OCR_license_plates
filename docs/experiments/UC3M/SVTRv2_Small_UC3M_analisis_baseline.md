# UC3M-LP — SVTRv2-Small: revisión cuantitativa del baseline zero-shot

## 1. Alcance y condiciones experimentales

- **Dataset:** UC3M-LP, partición oficial `test`; **491 recortes**; no se usa `train`.
- **GT parcial:** 488 etiquetas de longitud 7 y 3 de longitud 8; **dos posiciones anonimizadas (`*`) por matrícula**.
- **Modelo:** SVTRv2-Small (OpenOCR), `BGPD + GTCDecoder(infer_gtc=True)` y `GTCLabelDecode(only_gtc=True)`; checkpoint preentrenado **sin fine-tuning**.
- **Código OpenOCR:** commit `1ccfc6ee6161f7133b192e16af3a9265273997ba`.
- **SHA-256 del checkpoint:** `4c4e21c8eabd7a38519d9f10d6ffa286d470a07b1a8976109fe2b995a9414970`.
- **SHA-256 de la configuración:** `a22f090ee736cc8eddac6e6b2b5e7b7b8ae333c5ca5e74ba2dbfb2c4df55ee75`.
- **SHA-256 del manifiesto:** `97a6e041a996f2865796780f44b41ef4318b1c25066685e8293deb5df1939194`.
- **Recursos:** A100 40 GB, PyTorch `2.14.1+cu126`, Python `3.14.3`, batch size 8.

## 2. Resultados originales del script de evaluación

| Métrica | Resultado |
|---|---:|
| Imágenes evaluadas | **491/491** |
| Coincidencias parciales estrictas | **7/491 (1,43 %)** |
| Distancia de edición con comodín `*` | **991** |
| Caracteres conocidos | **2.458** |
| Tasa de error con máscara (*proxy*) | **40,32 %** |
| Predicciones de longitud completa correcta | **60/491 (12,22 %)** |
| Error posicional sobre caracteres conocidos **solo cuando la longitud coincide** | **139/300 (46,33 %)** |
| Tiempo de *GPU-forward* por imagen | **2,554 ms** |
| Parámetros | **12.518.159** |

**Interpretación:** el *EM parcial estricto* exige que la predicción tenga la longitud **total** original (incluidas dos posiciones ocultas) y coincida con todos los caracteres conocidos. La tasa de edición con máscara aplica una distancia Levenshtein con coste de sustitución nulo sobre `*`, pero sigue penalizando inserciones y eliminaciones. **No son EM/CER convencionales ni se comparan directamente con los valores de ELPD.**

## 3. Diagnóstico sobre caracteres visibles

Se compara la predicción normalizada con `ground_truth_visible`, cadena que contiene solo los caracteres conocidos, conservando su orden pero omitiendo las posiciones con `*`.

| Indicador auxiliar | Resultado |
|---|---:|
| Texto visible exactamente coincidente | **150/491 (30,55 %)** |
| Discrepancias frente al texto visible | **341/491 (69,45 %)** |
| Distancia Levenshtein acumulada | **515** |
| Tasa de edición del texto visible | **515/2.458 = 20,95 %** |
| Predicciones más largas que el texto visible | **317** |
| Predicciones de igual longitud que el texto visible | **161** |
| Predicciones más cortas que el texto visible | **13** |
| Predicción vacía | **1** |

**Advertencia:** una predicción de 7 caracteres puede ser correcta en las cinco posiciones conocidas y, sin embargo, presentar dos caracteres adicionales respecto a la cadena visible de cinco caracteres. Por tanto, **estas son métricas diagnósticas de transcripción visible y no prueban que los caracteres adicionales sean errores reales**. Tampoco se pueden interpretar como EM/CER de matrícula completa.

## 4. Tipología de las 341 discrepancias visibles

Se utiliza una alineación Levenshtein óptima. En caso de empate, se priorizan sustituciones, después inserciones y eliminaciones; la asignación individual de operaciones puede cambiar en otros alineamientos igualmente óptimos.

| Tipo de discrepancia | Matrículas |
|---|---:|
| Inserciones | 291 |
| Inserciones + Sustituciones | 26 |
| Sustituciones | 10 |
| Inserciones + Eliminaciones | 5 |
| Sustituciones + Eliminaciones | 3 |
| Inserciones + Sustituciones + Eliminaciones | 3 |
| Eliminaciones | 3 |
| **Total de discrepancias** | **341** |

Operaciones acumuladas frente al texto visible: **425 inserciones, 53 sustituciones y 37 eliminaciones**, en total **515**.

En **178 imágenes**, la predicción termina con todo el texto visible y contiene caracteres adicionales delante. Es un patrón descriptivo; **no demuestra** que esos prefijos sean erróneos, debido a los caracteres anonimizados.

### Casos para inspección visual

| Imagen | GT parcial | GT visible | Predicción | Distancia visible |
|---|---|---|---|---:|
| `test__00000__lp00.png` | `597*LK*` | `597LK` | `4597LK` | 1 |
| `test__00014__lp00.png` | `07*6*YR` | `076YR` | `B076YR` | 1 |
| `test__00015__lp00.png` | `4*73B*W` | `473BW` | `15473BW` | 2 |
| `test__01515__lp00.png` | `543*FZ*` | `543FZ` | `185431221` | 6 |
| `test__00606__lp01.png` | `624*GX*` | `624GX` | `∅` | 5 |
| `test__00818__lp00.png` | `7*03C*M` | `703CM` | `D` | 5 |
| `test__01454__lp00.png` | `*652F*M` | `652FM` | `L` | 5 |
| `test__01684__lp00.png` | `0*51*FN` | `051FN` | `MOSTEFN` | 5 |

- `test__00606__lp01.png`: **predicción vacía**.
- `test__00818__lp00.png` y `test__01454__lp00.png`: **truncamientos graves** respecto a los caracteres conocidos.
- `test__01515__lp00.png`: predicción con **múltiples discrepancias**, candidata a revisión de recorte/decodificación.
- `test__00000__lp00.png`: `4597LK` contiene `597LK`, pero añade un carácter inicial respecto al texto visible; **no se puede confirmar si ese carácter es incorrecto** sin analizar la región oculta.

## 5. Comparación con PARSeq y SVTRv2-Tiny (mismo test)

| Indicador | PARSeq | SVTRv2-Tiny | SVTRv2-Small |
|---|---:|---:|---:|
| EM parcial estricto | 1,63 % | **2,44 %** | 1,43 % |
| Error con máscara (*proxy*) ↓ | **39,83 %** | 43,25 % | 40,32 % |
| Predicciones de longitud total correcta | 26/491 | **107/491** | 60/491 |
| Coincidencia exacta de texto visible (*diagnóstico*) ↑ | **358/491 (72,91 %)** | 55/491 (11,20 %) | 150/491 (30,55 %) |
| Edición de texto visible (*diagnóstico*) ↓ | **8,62 %** | 30,92 % | 20,95 % |
| GPU-forward/imagen | 2,885 ms | **2,408 ms** | 2,554 ms |

### Aciertos de texto visible cruzados

| Comparación | Ambos aciertan | Solo Small acierta | Solo el otro acierta | Ambos fallan |
|---|---:|---:|---:|---:|
| Frente a Tiny | 36 | 114 | 19 | 322 |
| Frente a PARSeq | 126 | 24 | 232 | 109 |

Estas cifras reflejan únicamente el **criterio de texto visible**, condicionado por las posiciones ocultas. Small acierta más textos visibles que Tiny, pero no supera a PARSeq en ese diagnóstico. El EM parcial estricto no debe emplearse como único ranking de OCR cuando el GT incluye posiciones ocultas.

## 6. Conclusiones y pendientes

1. **SVTRv2-Small mejora claramente a Tiny** en transcripción diagnóstica del texto visible (30,55 % frente a 11,20 %), y reduce la distancia de edición visible.
2. **PARSeq mantiene el mejor rendimiento auxiliar** sobre los caracteres visibles (72,91 %), pero los tres modelos producen muy pocas predicciones con longitud completa correcta, condicionando el EM parcial estricto.
3. La discrepancia dominante de Small frente al texto visible es la **inserción** (aunque algunas pueden ocupar posiciones ocultas); existen además sustituciones, truncamientos y una salida vacía.
4. Para atribuir las causas a máscaras, iluminación, geometría o fallos del decodificador son necesarias **hojas de contacto y revisión visual**, no únicamente los CSV.
5. Antes de cerrar la evaluación formal de UC3M-LP, se debe validar un protocolo de alineamiento que contemple posiciones ocultas **opcionales en la salida**, separando recuperación de texto observable y reconstrucción de posiciones enmascaradas.
6. Las cifras de latencia registran solo *GPU-forward*, no todo el pre/postprocesado. Para comparar costes de forma definitiva se necesita un benchmark homogéneo.

**Estado:** baseline cuantitativo comprobado; revisión visual y protocolo final de evaluación parcial pendientes.
