# UC3M-LP — SVTRv2-Tiny: análisis del baseline zero-shot

## 1. Alcance y protocolo

- **Dataset:** UC3M-LP; partición oficial `test`; **491 recortes**.
- **Modelo:** SVTRv2-Tiny, OpenOCR, arquitectura `BGPD`, `GTCDecoder(infer_gtc=True)`, `GTCLabelDecode(only_gtc=True)`.
- **OpenOCR commit:** `1ccfc6ee6161f7133b192e16af3a9265273997ba`.
- **Checkpoint SHA-256:** `684f23bc3a2b1f724cadfd275e1a185456dd51b5e56c8830780d1998e90f77f6`.
- **Manifiesto SHA-256:** `97a6e041a996f2865796780f44b41ef4318b1c25066685e8293deb5df1939194`.
- **Condiciones:** zero-shot, sin fine-tuning; 1 GPU A100; batch size 8.
- **Ground truth:** 488 matrículas con 7 posiciones y 3 con 8; exactamente dos posiciones anonimizadas en las 491 anotaciones.

## 2. Resultados oficiales del baseline

| Métrica | Resultado |
|---|---:|
| Imágenes evaluadas | 491/491 |
| Coincidencias parciales estrictas | 12/491 (**2,44 %**) |
| Distancia de edición con máscara | 1.063 |
| Caracteres conocidos | 2.458 |
| Tasa de error con máscara (*proxy*) | **43,25 %** |
| Predicciones con longitud de matrícula completa correcta | 107/491 (**21,79 %**) |
| Errores sobre posiciones conocidas *solo* en longitudes correctas | 263/535 (**49,16 %**) |
| Tiempo GPU-forward por imagen | **2,408 ms** |
| Parámetros | **5.710.751** |

**Interpretación:** el EM parcial estricto exige longitud total idéntica y coincidencia en las posiciones conocidas. La tasa de edición con máscara utiliza `*` como comodín en la alineación; **no equivale al CER convencional de ELPD**.

## 3. Diagnóstico adicional sobre el texto visible

Para entender cómo afecta la anonimización, se compara la predicción normalizada con `ground_truth_visible`, una cadena formada exclusivamente por los caracteres conocidos en su orden original.

| Indicador diagnóstico | Resultado |
|---|---:|
| Coincidencias exactas de texto visible | 55/491 (**11,20 %**) |
| Discrepancias con texto visible | 436/491 |
| Distancia Levenshtein acumulada respecto al texto visible | **760** |
| Tasa de edición respecto al texto visible | **30,92 %** (760/2.458) |
| Predicciones más largas que el texto visible | 404 |
| Predicciones tan largas como el texto visible | 72 |
| Predicciones más cortas que el texto visible | 15 |

**Advertencia esencial:** esta comparación es un *diagnóstico*, **no una medida objetiva de todos los errores reales del modelo**. Penaliza la aparición de caracteres en posiciones anonimizadas que no pueden validarse. Por ejemplo, una predicción de 7 caracteres puede coincidir en las cinco posiciones visibles y obtener EM parcial estricto correcto, pero no coincidir con la cadena visible de 5 caracteres. De hecho, las **12 coincidencias parciales estrictas no pertenecen a las 55 coincidencias exactas de texto visible**. Son criterios diferentes, no resultados acumulables.

## 4. Clasificación de las discrepancias respecto al texto visible

Se ha utilizado un alineamiento de Levenshtein mínimo; en empates se prioriza sustitución, inserción y eliminación. La tipología puede variar en alineamientos igualmente óptimos.

| Categoría | Imágenes |
|---|---:|
| Solo inserciones | 367 |
| Inserciones + sustituciones | 37 |
| Inserciones + eliminaciones | 14 |
| Solo sustituciones | 7 |
| Solo eliminaciones | 5 |
| Inserciones + eliminaciones + sustituciones | 5 |
| Eliminaciones + sustituciones | 1 |
| **Total de discrepancias** | **436** |

Operaciones acumuladas frente al texto visible: **639 inserciones**, **71 sustituciones** y **50 eliminaciones** (**760** en total). **423 de las 436 discrepancias** incluyen al menos una inserción según este diagnóstico.

Estas inserciones **no deben atribuirse automáticamente a falsos caracteres**: dos posiciones del GT están ocultas y algunos caracteres generados pueden ocuparlas. Se requiere inspección visual para distinguir espacios anonimizados, elementos gráficos y errores de decodificación.

### Ejemplos que conviene inspeccionar

| Imagen | GT parcial | GT visible | Predicción | Observación |
|---|---|---|---|---|
| `test__00015__lp00.png` | `4*73B*W` | `473BW` | `8473BW` | Carácter inicial adicional frente al texto visible |
| `test__00035__lp00.png` | `72*7*PP` | `727PP` | `8727PP` | Carácter inicial adicional |
| `test__00000__lp00.png` | `597*LK*` | `597LK` | `185970LK7` | Caracteres adicionales / alineamiento más difícil |
| `test__00393__lp00.png` | `R*366BD*` | `R366BD` | `R` | Truncamiento grave |
| `test__00311__lp00.png` | `*767*BW` | `767BW` | `B` | Truncamiento grave |
| `test__00188__lp00.png` | `8*14G*Y` | `814GY` | `8814GGY` | **EM parcial estricto correcto**, pero no texto visible idéntico |

En **216** imágenes, el texto visible completo aparece al final de la predicción, precedido por caracteres adicionales. Los prefijos adicionales más frecuentes en estos casos son `1`, `8` y `4`. Se trata de un **patrón descriptivo**, no de una explicación causal.

## 5. Comparación preliminar con PARSeq (mismo test)

| Métrica | PARSeq | SVTRv2-Tiny |
|---|---:|---:|
| EM parcial estricto | 1,63 % (8/491) | 2,44 % (12/491) |
| Tasa de error con máscara | **39,83 %** | 43,25 % |
| Longitud completa correcta | 26/491 | 107/491 |
| Coincidencia exacta del texto visible (*diagnóstico*) | **358/491 (72,91 %)** | 55/491 (11,20 %) |
| Edición del texto visible (*diagnóstico*) | **8,62 %** | 30,92 % |
| GPU-forward/imagen | 2,885 ms | 2,408 ms |

Con el criterio de texto visible, ambos aciertan 46 imágenes, **solo PARSeq acierta 312**, **solo Tiny acierta 9** y ambos fallan 124. El criterio de texto visible está condicionado por las posiciones ocultas, por lo que **estos datos no permiten concluir una diferencia equivalente de OCR sobre matrículas completas**.

## 6. Conclusiones y siguientes pasos

1. La evaluación parcial estricta es **muy sensible a la longitud total prevista**, que no siempre coincide con la salida de un OCR aplicado a placas anonimizadas.
2. Tiny genera con frecuencia **predicciones más largas que el texto visible**. Se debe separar el reconocimiento de caracteres conocidos de la generación de caracteres en huecos ocultos.
3. Aparecen **truncamientos graves** en algunos recortes y, en otros, cadenas con inserciones o posibles confusiones de caracteres.
4. Para investigar las causas necesitamos **hojas de contacto con imagen, GT parcial, texto visible y predicción**. No cabe atribuir los fallos a perspectiva, máscara o elementos gráficos solo a partir del CSV.
5. El protocolo de benchmarking de latencia todavía debe unificarse (misma carga, *warm-up*, batch, pre/postprocesado) antes de obtener conclusiones de eficiencia entre arquitecturas.
6. Debemos mantener **las métricas originales y las diagnósticas separadas**, y aplicar el mismo protocolo a PARSeq, SVTRv2-Small y MDiff4STR-Small.

**Estado:** análisis cuantitativo completado; inspección visual de las discrepancias pendiente.
