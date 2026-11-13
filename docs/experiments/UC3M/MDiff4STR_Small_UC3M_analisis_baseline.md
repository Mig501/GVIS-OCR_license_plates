# UC3M-LP — MDiff4STR-Small: análisis del baseline zero-shot

## 1. Alcance y reproducibilidad

- **Dataset:** UC3M-LP, partición oficial `test`; **491 recortes** (de 2.547 recortes totales). No se utiliza `train` para el baseline.
- **Ground truth:** las 491 etiquetas son parciales; cada una contiene exactamente **2 posiciones anonimizadas (`*`)**. Hay 488 etiquetas de longitud 7 y 3 de longitud 8.
- **Modelo:** MDiff4STR-Small, sin fine-tuning; decodificación `MDiffDecoder`, **3 pasos**, `sample_k=3`, `semi_autoregressive=True`, semilla 42.
- **Checkpoint SHA-256:** `ebc96f11e5739eae98491f7c8c140157ed12b35dafa220511ef9d399cc20a937`.
- **Configuración SHA-256:** `0abdd478db64fa06b0574f56f8c8620800bab511af0af94861b9ea73bcb219bf`.
- **OpenOCR commit:** `1ccfc6ee6161f7133b192e16af3a9265273997ba`.
- **Manifiesto SHA-256:** `97a6e041a996f2865796780f44b41ef4318b1c25066685e8293deb5df1939194`.
- **Entorno:** NVIDIA A100-PCIE-40GB, PyTorch 2.14.1+cu126, Python 3.14.3, batch size 8; **18.891.072 parámetros**.

## 2. Resultados originales del script

| Indicador | Resultado |
|---|---:|
| Imágenes evaluadas | **491/491** |
| EM parcial estricto | **21/491 (4,28 %)** |
| Distancia de edición con máscara `*` | **934** |
| Caracteres GT conocidos | **2.458** |
| Tasa de edición con máscara (*proxy*) | **38,00 %** |
| Predicciones con longitud total correcta | **71/491 (14,46 %)** |
| Errores posicionales sobre conocidos, **solo cuando coincide la longitud** | **119/355 (33,52 %)** |
| Tiempo GPU-forward/imagen | **2,629 ms** |
| Tiempo inferencia con carga/preparación de imágenes | **4,546 s** |

**Interpretación del protocolo original:** `partial_EM` exige longitud completa de etiqueta y coincidencia en todas las posiciones conocidas; la tasa de edición con máscara utiliza `*` como comodín de coste cero para una sustitución, pero penaliza inserciones y eliminaciones. Estas métricas **no equivalen al EM/CER completo de ELPD**. La evaluación sobre longitudes coincidentes se calcula únicamente en un subconjunto condicionado y no debe interpretarse como rendimiento sobre todo UC3M-LP.

## 3. Diagnóstico adicional: comparación con texto visible

Se compara cada `prediction_normalized` con `ground_truth_visible`, formado al eliminar `*` de la etiqueta conservando el orden de los caracteres observables. Recalculado a partir de las 491 filas del CSV.

| Indicador auxiliar | Resultado |
|---|---:|
| Coincidencias exactas del texto visible | **87/491 (17,72 %)** |
| Discrepancias frente al texto visible | **404/491 (82,28 %)** |
| Distancia de Levenshtein acumulada | **590** |
| Tasa de edición respecto al texto visible | **24,00 % (590/2.458)** |
| Predicciones más largas que el visible | **369** |
| Predicciones de igual longitud que el visible | **110** |
| Predicciones más cortas que el visible | **12** |
| Predicciones vacías | **0** |
| Predicciones que terminan con el visible íntegro y añaden un prefijo | **155** |

**Advertencia metodológica fundamental:** los dos `*` representan posiciones ocultas. Si el OCR genera un carácter en una de esas posiciones, **no es necesariamente un error**, aunque se contabilice como inserción al comparar con la cadena de texto visible reducida. El indicador de texto visible es **diagnóstico**, no una métrica validada de reconocimiento de matrícula completa ni directamente comparable con ELPD. A su vez, el EM parcial estricto penaliza las salidas que omiten posiciones ocultas aunque lean todo lo visible.

## 4. Tipología de las 404 discrepancias con el texto visible

Se descompone cada distancia de Levenshtein mínima priorizando, ante empate, sustitución, inserción y eliminación. Los tipos asignados a cada ejemplo pueden variar con otras alineaciones óptimas.

| Categoría de discrepancia | Matrículas |
|---|---:|
| Inserciones | 327 |
| Inserciones + Sustituciones | 42 |
| Sustituciones | 17 |
| Inserciones + Eliminaciones | 9 |
| Eliminaciones | 5 |
| Sustituciones + Eliminaciones | 4 |
| **Total** | **404** |

Operaciones acumuladas en el diagnóstico: **472 inserciones, 82 sustituciones y 36 eliminaciones** (590 en total). Las inserciones describen diferencias frente a la **cadena visible**, y no deben clasificarse sin más como caracteres falsos debido a la anonimización.

### Casos para revisión visual

| Imagen | GT parcial | GT visible | Predicción | Distancia visible | Motivo de selección |
|---|---|---|---|---:|---|
| `test__00771__lp00.png` | `336*LS*` | `336LS` | `0` | 5 | Truncamiento grave |
| `test__00818__lp00.png` | `7*03C*M` | `703CM` | `L` | 5 | Truncamiento grave |
| `test__00979__lp00.png` | `5*30*FF` | `530FF` | `TB30FFFF` | 4 | Repetición de caracteres |
| `test__00000__lp00.png` | `597*LK*` | `597LK` | `4597LKK` | 2 | Prefijo y repetición frente al visible |
| `test__00015__lp00.png` | `4*73B*W` | `473BW` | `5473BW` | 1 | Prefijo frente al visible |
| `test__00867__lp01.png` | `92*2FP*` | `922FP` | `89202FPII` | 4 | Inserciones y sustituciones |
| `test__00591__lp03.png` | `*999*YM` | `999YM` | `9999YM` | 1 | Repetición con confianza alta |

**Hallazgos observables en las predicciones:** existen salidas muy cortas (`0`, `L` o `7` para etiquetas visibles de cinco caracteres), repeticiones e inserciones de prefijos/sufijos. **No se observan salidas vacías** en este baseline de UC3M-LP. Los truncamientos respecto al texto conocido sí aparecen, a diferencia de la ausencia de truncamientos graves descrita para el conjunto ELPD en la revisión visual anterior.

No es posible atribuir las causas a desenfoque, máscara, perspectiva o decodificación **sin inspeccionar el recorte correspondiente**.

## 5. Comparación con los cuatro baselines (mismo test)

| Modelo | EM parcial estricto ↑ | Error con máscara (proxy) ↓ | Longitud total correcta | Exactitud texto visible (diagnóstico) ↑ | Edición texto visible (diagnóstico) ↓ | GPU-forward/imagen |
|---|---:|---:|---:|---:|---:|---:|
| PARSeq | 8/491 (1,63 %) | 39,83 % | 26/491 | 358/491 (72,91 %) | 8,62 % | 2,885 ms |
| SVTRv2-Tiny | 12/491 (2,44 %) | 43,25 % | 107/491 | 55/491 (11,20 %) | 30,92 % | 2,408 ms |
| SVTRv2-Small | 7/491 (1,43 %) | 40,32 % | 60/491 | 150/491 (30,55 %) | 20,95 % | 2,554 ms |
| MDiff4STR-Small | 21/491 (4,28 %) | 38,00 % | 71/491 | 87/491 (17,72 %) | 24,00 % | 2,629 ms |

**El orden cambia según el protocolo:** MDiff4STR-Small obtiene el mejor EM parcial estricto y la menor tasa de edición con máscara, pero solo 87 coincidencias exactas con el texto visible, menos que PARSeq y SVTRv2-Small. Por ello, **no se debe afirmar que sea el mejor OCR de UC3M-LP usando únicamente las métricas parciales originales**.

### Comparación por imagen: coincidencia exacta del texto visible

| Frente a | Ambos aciertan | Solo MDiff4STR acierta | Solo el otro acierta | Ambos discrepan |
|---|---:|---:|---:|---:|
| PARSeq | 69 | 18 | 289 | 115 |
| SVTRv2-Tiny | 19 | 68 | 36 | 368 |
| SVTRv2-Small | 38 | 49 | 112 | 292 |

Esta tabla solo mide el criterio auxiliar de `ground_truth_visible`: **no prueba que el modelo con más coincidencias visibles reconstruya mejor las dos posiciones anonimizadas**.

## 6. Conclusiones y próximos pasos

1. **MDiff4STR-Small completa el cuarto baseline de UC3M-LP** y presenta el mejor resultado entre los cuatro con las dos métricas parciales originales (EM parcial estricto y error con máscara).
2. La **anonimización distorsiona la interpretación**: comparado con texto visible, MDiff4STR-Small acierta 87/491, mientras PARSeq acierta 358/491 y SVTRv2-Small 150/491. Tampoco este diagnóstico es una evaluación definitiva sin resolver la alineación de posiciones ocultas.
3. El patrón más común de discrepancia con el texto visible son las **inserciones**. Se observan también sustituciones, repeticiones y truncamientos graves en algunos casos; no hay predicciones vacías.
4. Para una revisión causal, generar hojas de contacto con **imagen, GT parcial, GT visible, predicción y confianza**, priorizando truncamientos, discrepancias de alta confianza y muestras con múltiples errores.
5. **Definir y validar una métrica parcial con alineación tolerante a posiciones ocultas opcionalmente omitidas** antes de usar el rendimiento UC3M-LP para seleccionar el modelo ganador. Conservar siempre las métricas originales para reproducibilidad.
6. Unificar el benchmark de latencia: los tiempos aquí registrados cubren solo *GPU-forward*, no el pre/postprocesado completo.

**Estado:** métricas originales y diagnósticas recalculadas; revisión visual y validación del protocolo definitivo de UC3M-LP pendientes.
