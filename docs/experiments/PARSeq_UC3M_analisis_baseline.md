# UC3M-LP — Revisión del baseline zero-shot PARSeq

## 1. Alcance y reproducibilidad

- Dataset: **UC3M-LP**, partición oficial `test` con **491 recortes LPO**; no se utiliza `train`.
- Anotaciones: **491 etiquetas parciales**; todas tienen **2 posiciones anonimizadas (`*`)**. 488 etiquetas tienen longitud 7 y 3 tienen longitud 8.
- Modelo: PARSeq preentrenado, sin fine-tuning, modo autorregresivo más una iteración de refinamiento.
- Checkpoint: `parseq-bb5792a6.pt`.
- SHA-256 checkpoint: `bb5792a68e367476abca029cbf8699abc805f3d3dc7e57aae45c8ec4f7b7cd00`.
- SHA-256 manifiesto: `97a6e041a996f2865796780f44b41ef4318b1c25066685e8293deb5df1939194`.
- GPU: NVIDIA A100-PCIE-40GB; batch 8; resolución 32×128.

## 2. Métricas registradas por el script original

| Métrica | Resultado |
|---|---:|
| EM parcial estricto | **1,63 % (8/491)** |
| Predicciones de la longitud original (7 u 8) | **26/491 (5,30 %)** |
| Tasa de edición con `*` como comodín (proxy) | **39,83 % (979/2.458)** |
| Error en posiciones conocidas condicionado a igual longitud | **27,69 % (36/130)** |
| GPU-forward/imagen | **2,885 ms** |

**Advertencia metodológica:** el EM parcial estricto requiere que la predicción tenga la longitud completa original; esto penaliza las transcripciones de solo los cinco o seis caracteres visibles. La tasa de edición con comodín también incluye penalizaciones por las dos posiciones anonimizadas cuando el OCR las omite. No debe interpretarse el 39,83 % como tasa convencional de caracteres visualmente reconocidos de forma incorrecta.

## 3. Cálculo diagnóstico adicional: caracteres visibles

Para diagnosticar cómo reconoce el modelo el texto observable, se compara `prediction_normalized` con `ground_truth_visible` (el GT sin `*`). Es una **métrica auxiliar de transcripción visible**, no una medida de matrícula completa y **no comparable directamente con ELPD**.

| Métrica auxiliar | Resultado |
|---|---:|
| Coincidencia exacta del texto visible | **358/491 = 72,91 %** |
| Muestras con alguna diferencia respecto al texto visible | **133/491 = 27,09 %** |
| Distancia de Levenshtein acumulada respecto al texto visible | **212** |
| Tasa de edición respecto al texto visible | **212/2.458 = 8,62 %** |
| Predicciones de longitud igual al número de caracteres conocidos | **374/491 (76,17 %)** |

**Ejemplo determinante:** `597*LK*` tiene texto visible `597LK` y PARSeq predice `597LK`. El script original marca EM parcial estricto = 0 y distancia con comodines = 2, pese a que los cinco caracteres observables coinciden.

## 4. Clasificación de los 133 desaciertos sobre el texto visible

| Categoría | Imágenes |
|---|---:|
| Inserciones | 85 |
| Sustituciones | 16 |
| Eliminaciones | 14 |
| Inserciones + sustituciones | 10 |
| Sustituciones + eliminaciones | 7 |
| Inserciones + eliminaciones | 1 |
| **Total** | **133** |

En una alineación óptima escogida para cada par de textos se contabilizan **136 inserciones, 44 sustituciones y 32 eliminaciones**. La clasificación de cada operación puede variar cuando hay varias alineaciones Levenshtein igualmente óptimas; la distancia total no cambia.

### Casos destacados

| GT anonimizado | Texto visible | Predicción | Distancia visible |
|---|---|---|---:|
| `71*9GH*` | `719GH` | `4719GHIIII` | 5 |
| `7*03C*M` | `703CM` | `0` | 4 |
| `505**WW` | `505WW` | `5050WWWW` | 3 |
| `686**MM` | `686MM` | `686PMMMM` | 3 |
| `22*2*DY` | `222DY` | `72207` | 3 |
| `2*28L*L` | `228LL` | `28LL` | 1 |

Las repeticiones y caracteres añadidos son observables en las cadenas. Las **causas visuales** (desenfoque, perspectiva, regiones de anonimización, elementos de la placa) **no pueden establecerse sin revisar las imágenes originales**.

## 5. Interpretación y pasos recomendados

1. **Mantener** las métricas originales del baseline como información reproducible, pero **no utilizarlas aisladas** para juzgar la calidad de lectura de caracteres visibles.
2. Incorporar la **coincidencia exacta y tasa de edición del texto visible** como diagnósticos de UC3M-LP. Son susceptibles de penalizar hipotéticas predicciones en posiciones anonimizadas; deben presentarse explícitamente como métricas auxiliares.
3. Aplicar **el mismo protocolo a los `predictions.csv` de SVTRv2-Tiny, SVTRv2-Small y MDiff4STR-Small**, sin volver a ejecutar GPU; de lo contrario, la comparación intra-dataset podría resultar engañosa.
4. Para una evaluación más rigurosa, diseñar y documentar un **alineamiento tolerante a posiciones desconocidas y opcionalmente omitidas**, verificándolo sobre casos anotados antes de usarlo como métrica principal.
5. Revisar visualmente los **133 desaciertos sobre texto visible**, priorizando predicciones largas/repetidas, truncamientos y discrepancias de alta confianza.

**Estado:** baseline técnico ejecutado; análisis cuantitativo revisado; causas visuales y protocolo final de métricas parciales aún por validar.
