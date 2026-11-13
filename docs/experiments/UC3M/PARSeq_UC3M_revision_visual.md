# UC3M-LP — Revisión visual de errores del baseline PARSeq

**Modelo:** PARSeq · **Evaluación:** zero-shot, sin fine-tuning · **Partición:** test, 491 recortes · **Fuente:** `error_analysis.zip` (hojas de contacto, `all_predictions_review.csv`, `errors.csv` y `error_analysis.md`).

## 1. Protocolo y cautelas

- El *ground truth* (GT) de UC3M-LP contiene posiciones anonimizadas con `*`. Los rectángulos ocultan el carácter original: **no sabemos qué letra o número había debajo**.
- Se inspeccionaron las **nueve hojas de contacto** entregadas: cinco hojas generales (60 ejemplos, 12 por hoja) y cuatro hojas temáticas (48 ejemplos, posiblemente solapados). La revisión visual es **selectiva**, no exhaustiva de los 491 recortes ni de las 133 discrepancias.
- Las categorías `insertion`, `deletion` y `substitution` de los CSV provienen del alineamiento con el **texto visible** (`*` eliminados). Por sí solas **no demuestran un error visual de OCR**: un carácter adicional podría corresponder a una posición anonimizada. Las métricas estrictas con máscara y las del texto visible responden a preguntas diferentes.

## 2. Métricas verificadas

| Indicador | Resultado |
|---|---:|
| Recortes evaluados | 491 |
| EM parcial estricto (requiere longitud de GT original) | **1,63 % (8/491)** |
| Distancia de edición con máscara / caracteres conocidos (proxy; no CER) | **39,83 % (979/2458)** |
| Predicciones con longitud del GT original (7) | **26/491** |
| Coincidencia exacta con texto visible (diagnóstico) | **72,91 % (358/491)** |
| Tasa de edición frente a texto visible (diagnóstico) | **8,62 % (212/2458)** |
| Discrepancias frente a texto visible | **133/491** |
| Predicciones más largas / más cortas que texto visible | **95 / 22** |
| Predicciones vacías | **0** |

**Clasificación automática de las 133 discrepancias visibles:** solo inserciones 85; solo sustituciones 16; solo eliminaciones 14; mixtas 18. Son **operaciones de comparación**, no motivos de fallo confirmados.

## 3. Hallazgos de la inspección visual

### 3.1. Truncamientos en matrículas de dos líneas (confirmados)

El modelo puede terminar su predicción después de leer solo una parte de la matrícula, omitiendo caracteres visibles de la segunda fila. Es un patrón especialmente claro en placas de formato cuadrado o con texto distribuido verticalmente.

| Imagen | GT parcial | Predicción | Lo que se observa |
|---|---|---|---|
| `test__00818__lp00.png` | `7*03C*M` | `0` | Matrícula de dos filas; omite caracteres visibles de ambas zonas. |
| `test__00817__lp01.png` | `36*7KW*` | `367` | Se distinguen `K` y `W` en la fila inferior, pero no aparecen en la salida. |
| `test__00771__lp00.png` | `336*LS*` | `3361` | Omite `LS`, visible en la fila inferior; confianza indicada: **0,872**. |
| `test__00376__lp00.png` | `01*6H*J` | `2016` | Las letras `H` y `J` de la fila inferior desaparecen. |

La estructura de dos líneas es una **característica común a estos casos**, no una causa demostrada por un experimento controlado.

### 3.2. Truncamientos en matrículas de una línea (confirmados)

| Imagen | GT parcial | Predicción | Lo que se observa |
|---|---|---|---|
| `test__01546__lp00.png` | `694**XM` | `694` | Omite el bloque visible `XM` pese a estar presente en el recorte; confianza **0,737**. |
| `test__00901__lp01.png` | `949**TS` | `949` | Omite `TS`, visible a la derecha. |

Estos casos impiden atribuir todos los truncamientos exclusivamente a la disposición en dos filas.

### 3.3. Repeticiones e inserciones de caracteres (con distinto grado de certeza)

| Imagen | GT parcial | Predicción | Interpretación |
|---|---|---|---|
| `test__00151__lp01.png` | `505**WW` | `5050WWWW` | Predicción de **8 caracteres frente a 7 posiciones**; hay al menos una inserción respecto de la longitud original, y repetición aparente de `W`. |
| `test__00511__lp00.png` | `686**MM` | `686PMMMM` | Predicción de **8 caracteres**; hay al menos una inserción y repetición aparente de `M`. |
| `test__00132__lp00.png` | `036*DR*` | `036DRDR` | Repite aparentemente `DR`, pero sin conocer las posiciones tapadas no se puede reconstruir la cadena verdadera. |
| `test__00039__lp00.png` | `733**JW` | `73333JW` | **No etiquetar como error confirmado**: la predicción tiene longitud 7 y encaja con todos los caracteres conocidos (`Dmask=0`); los dos `3` adicionales ocupan posiciones ocultas. |

**Regla para la revisión:** diferenciar la **repetición visualmente aparente** de una **inserción demostrada**. Las máscaras impiden saber cuáles eran los caracteres verdaderos en esas posiciones.

### 3.4. Errores con confianza elevada (confirmados en posiciones visibles)

| Imagen | GT parcial | Predicción | Confianza | Observación |
|---|---|---|---:|---|
| `test__00211__lp01.png` | `2*28L*L` | `28LL` | **0,940** | Se pierde un `2` conocido. |
| `test__00771__lp00.png` | `336*LS*` | `3361` | **0,872** | Se omiten `LS`, visibles en la segunda fila. |
| `test__00817__lp01.png` | `36*7KW*` | `367` | **0,807** | Se omiten `KW`, visibles en la segunda fila. |

La confianza es un **score del modelo**, no una probabilidad calibrada de que la predicción sea correcta. Estos ejemplos justifican no usar un umbral de confianza como único filtro de calidad.

### 3.5. Calidad visual y encuadre (asociaciones observadas)

Entre los recortes con discrepancias aparecen imágenes **muy oscuras**, **desenfocadas**, **sobreexpuestas**, con **perspectiva** o **resolución efectiva reducida**; por ejemplo, `test__00393__lp01.png` y `test__00591__lp01.png` dificultan apreciar varios caracteres. La existencia de estas condiciones es observable, pero no podemos atribuir de forma exclusiva el error a cada factor sin pruebas adicionales.

Las hojas muestran también **rectángulos grises de anonimización** que interrumpen la cadena. **No se ha verificado** que PARSeq los interprete como caracteres concretos ni como escudos: no debe trasladarse automáticamente a UC3M-LP esa hipótesis procedente de ELPD.

## 4. Interpretación metodológica

- **358 predicciones (72,91 %) reproducen exactamente el texto no anonimizado**, pero no deben contarse como 358 matrículas completas correctas: el criterio solo comprueba los caracteres visibles en una cadena condensada.
- **8/491** cumplen el EM parcial estricto, una condición especialmente exigente cuando el modelo no produce explícitamente las posiciones ocultas.
- Las 85 discrepancias clasificadas automáticamente como «solo inserciones» respecto del texto visible **no son 85 inserciones OCR demostradas**.
- En un mismo caso se pueden observar un error real de reconocimiento y una discrepancia artificial de longitud debida a la máscara.
- Conviene validar una métrica que permita que `*` represente una posición de carácter desconocido **o un hueco omitido en la salida**, y registrar separadamente la cobertura de caracteres visibles, longitud y errores por formato de placa. La definición exacta debe decidirse antes de recalcular la clasificación final de los cuatro modelos.

## 5. Notas breves para el tutor

- **Truncamientos pronunciados**, especialmente al procesar matrículas con caracteres en dos filas; también ocurren en placas de una fila.
- **Omisión de letras claramente visibles** al final de algunas matrículas (`336*LS* → 3361`, `694**XM → 694`).
- **Repeticiones de letras y números**, con algunas predicciones más largas que las siete posiciones originales (`505**WW → 5050WWWW`).
- **Errores con confianza elevada**, incluidos truncamientos relevantes (0,807–0,940 en ejemplos revisados).
- **Fallos asociados visualmente a mala iluminación, desenfoque y perspectiva**, sin causalidad establecida.
- **Precaución con la anonimización:** parte de las «inserciones» y del bajo EM parcial son consecuencias del modo de comparar con `*`, no errores necesariamente demostrados del reconocimiento.

## 6. Conclusión provisional

PARSeq muestra **buena transcripción de los cinco caracteres conocidos** bajo el diagnóstico de texto visible, pero la inspección identifica fallos reales de **truncamiento**, **omisión de letras**, **algunas repeticiones** y **errores de alta confianza**. Las posiciones anonimizadas condicionan mucho la evaluación automática: las conclusiones sobre número y tipo exacto de errores deben formularse con precaución. Esta revisión visual queda cerrada para PARSeq y puede usarse como plantilla para SVTRv2-Tiny, SVTRv2-Small y MDiff4STR-Small.
