# UC3M-LP — Revisión visual del baseline MDiff4STR-Small

## 1. Objetivo y alcance

Revisión visual y cuantitativa del modelo **MDiff4STR-Small**, sin fine-tuning, sobre los **491 recortes de test de UC3M-LP**, a partir del ZIP `error_analysis(3).zip` generado por `analyze_uc3m_baseline_errors.py`.

Se han examinado las **nueve hojas de contacto** proporcionadas (cinco generales y cuatro temáticas), así como las **491 filas** de `all_predictions_review.csv`, las **404 discrepancias** de `errors.csv` y el informe automático. La revisión visual es **selectiva**: las hojas de contacto no incluyen todas las matrículas, por lo que los ejemplos inspeccionados no representan un examen individual de las 491 imágenes.

**Precaución fundamental:** cada etiqueta parcial incluye posiciones anonimizadas con `*`; desconocemos sus caracteres reales. Una inserción frente a la cadena sin asteriscos no es, por sí sola, un error verificable de OCR.

## 2. Resultados cuantitativos

| Métrica original | Resultado |
|---|---:|
| Test procesado | 491 / 491 |
| EM parcial estricto (longitud completa y posiciones conocidas) | **4,28 % (21/491)** |
| Tasa de edición con máscara, *proxy* (no CER convencional) | **38,00 % (934/2458)** |
| Predicciones con longitud anotada correcta | 71/491 |
| GPU-forward por imagen (batch 8; no latencia extremo a extremo) | 2,629 ms |
| Parámetros | 18,89 M |

| Diagnóstico adicional | Resultado |
|---|---:|
| Predicciones iguales a la cadena visible | **87/491 (17,72 %)** |
| Predicciones distintas de la cadena visible | **404/491** |
| Distancia de edición frente a cadena visible (auxiliar) | **24,00 % (590/2458)** |
| Predicciones más largas que cadena visible | 369 |
| Predicciones más cortas que cadena visible | 12 |
| Cadena visible como sufijo estricto de la predicción | 155 |
| Predicciones vacías | **0** |
| Salidas de un único carácter | **3** |

Las métricas sobre texto visible son **diagnósticas y no equivalen** ni a reconocimiento completo de matrícula ni a EM/CER convencional. El hecho de que el texto conocido esté presente no valida los caracteres ocultos.

### Comprobaciones que reducen la ambigüedad de las máscaras

Se han añadido dos criterios conservadores, calculados sobre las 491 predicciones:

- **77 predicciones** no contienen la secuencia completa de caracteres conocidos en su orden anotado. Bajo la hipótesis habitual de conservar el orden de lectura, al menos un carácter conocido se ha omitido, alterado o desplazado de forma no explicable solo por los asteriscos.
- **11 predicciones** superan incluso la longitud **completa anotada** (no solo la longitud visible): son salidas excesivamente largas con independencia del contenido de las posiciones ocultas.
- La unión de ambos conjuntos comprende **87 predicciones** (hay solapamiento). Es una **cota conservadora de discrepancias confirmables a partir de la anotación**, no un recuento total de errores ni una métrica estándar de evaluación.

Estos criterios no sustituyen el examen visual individual y no permiten inferir las causas de los fallos.

## 3. Errores clave de la revisión visual

### 3.1. Truncamientos graves, especialmente en matrículas de dos líneas

**Observación visual confirmada.** En varios recortes con dos filas, el modelo devuelve un solo símbolo y pierde numerosos caracteres visibles. Aparecen además truncamientos en matrículas de una línea o degradadas.

| Imagen | GT parcial | Predicción | Observación |
|---|---|---|---|
| `test__00771__lp00.png` | `336*LS*` | `0` | Matrícula de dos líneas: pierde `336` y `LS` visibles; confianza 0,436. |
| `test__00818__lp00.png` | `7*03C*M` | `L` | Matrícula de dos líneas: prácticamente toda la lectura se pierde; confianza 0,377. |
| `test__00817__lp01.png` | `36*7KW*` | `7` | En formato de dos filas, omite `36` y `KW` visibles; confianza 0,302. |
| `test__00798__lp00.png` | `211**NG` | `821` | Lee parcialmente la primera fila y pierde el `NG` de la inferior. |
| `test__00358__lp00.png` | `999**TW` | `899` | Omite `TW` visible en la segunda línea. |

En estas imágenes la estructura de dos líneas es un factor **plausible** de dificultad; no se puede asegurar causalidad sin pruebas específicas.

### 3.2. Omisión o sustitución de caracteres conocidos

- `MA*117*S → MA11S` (`test__00533__lp00.png`): falta el `7` conocido aun cuando la confianza es **0,909**.
- `80*8*YF → 88` (`test__00104__lp00.png`): pierde letras `YF` y varios números conocidos; la imagen es oscura.
- `01*6H*J → E016` (`test__00376__lp00.png`): conserva algunos números iniciales, pero pierde `H` y `J` visibles; se aprecia un formato de dos líneas.
- `98*0*CY → 898OCY` (`test__00320__lp00.png`): **posible confusión visual `0/O`** en la posición conocida. El ejemplo justifica revisar confusiones de caracteres similares, sin atribuirles frecuencia general a partir de las hojas.

### 3.3. Repeticiones, prefijos y longitud excesiva

Se aprecian duplicaciones de caracteres y salidas con prefijos inesperados. La repetición es más clara cuando la predicción excede la longitud completa de la matrícula.

| Imagen | GT parcial | Predicción | Interpretación |
|---|---|---|---|
| `test__00979__lp00.png` | `5*30*FF` | `TB30FFFF` | Ocho caracteres para siete posiciones; repetición final y lectura de `5` alterada; imagen sobreexpuesta. |
| `test__00867__lp01.png` | `92*2FP*` | `89202FPII` | Nueve caracteres para siete posiciones; imagen muy desenfocada. |
| `test__01528__lp00.png` | `677**XV` | `16777XXV` | Nueve caracteres para siete posiciones; duplicaciones de números/letras y prefijo. |
| `test__00000__lp00.png` | `597*LK*` | `4597LKK` | Prefijo no justificado por una máscara inicial (el GT empieza por `597`), junto a `K` adicional. |

De las 155 predicciones con la cadena visible como sufijo estricto, los prefijos más comunes son **`8` (53 casos)**, **`1` (39)** y **`4` (13)**. Estos recuentos son descriptivos: un prefijo no demuestra por sí solo un error en todos los casos. No se ha acreditado en estas hojas un patrón dominante de escudos nacionales mal interpretados.

### 3.4. Fallos asociados a calidad visual y disposición

Entre las imágenes de mayor discrepancia hay ejemplos desenfocados (`test__00867__lp01.png`), sobreexpuestos (`test__00979__lp00.png`), oscuros (`test__00104__lp00.png`) y con inclinación o perspectiva (`test__00478__lp00.png`). Esto **coincide con** dificultades habituales de OCR, pero la muestra visual no demuestra que esos factores expliquen todos los fallos.

### 3.5. Confianza alta pese a errores comprobables

La confianza del modelo no equivale a una probabilidad calibrada de que la matrícula esté correctamente transcrita. En **4 predicciones con confianza ≥ 0,90**, el texto predicho ni siquiera contiene todos los caracteres conocidos en su orden, de acuerdo con el diagnóstico anterior.

Ejemplo revisado: `MA*117*S → MA11S` (confianza **0,909**). Otras salidas aparentemente correctas respecto a la cadena visible también presentan confianza alta, pero **no alcanzan el EM parcial estricto** por su longitud reducida.

## 4. Efecto de la anonimización y cautelas interpretativas

Un ejemplo es `0*61B*S → 061BS` (`test__01134__lp00.png`): la predicción contiene todos los caracteres visibles exactamente, pero el EM parcial estricto sigue siendo incorrecto por la diferencia de longitud. Este caso **no acredita un error de lectura de los caracteres conocidos**.

No debe deducirse que los caracteres adicionales frente al GT visible sean siempre falsos: las posiciones `*` podrían admitir parte de ellos. Del mismo modo, una predicción que coincide con todos los caracteres visibles no permite verificar que haya reconocido el texto de la matrícula completa.

La clasificación automática de inserciones/sustituciones/eliminaciones en `errors.csv` describe una alineación con el **texto visible reducido**, no causas visuales confirmadas. En esta ejecución presenta: 327 casos de solo inserciones, 17 de solo sustituciones, 5 de solo eliminaciones y 55 mixtos; los **404** suman todas las discrepancias visibles.

## 5. Comparación provisional de las cuatro arquitecturas en UC3M-LP

| Modelo | EM parcial estricto ↑ | Error con máscara (*proxy*) ↓ | Exactitud frente al texto visible (*auxiliar*) ↑ |
|---|---:|---:|---:|
| PARSeq | 1,63 % | 39,83 % | **72,91 %** |
| SVTRv2-Tiny | 2,44 % | 43,25 % | 11,20 % |
| SVTRv2-Small | 1,43 % | 40,32 % | 30,55 % |
| **MDiff4STR-Small** | **4,28 %** | **38,00 %** | 17,72 % |

**Interpretación:** MDiff4STR-Small encabeza las dos métricas parciales originales, mientras PARSeq reconoce con mayor frecuencia la cadena visible reducida. **Ningún criterio aislado establece el mejor OCR para matrículas completas**, pues las máscaras favorecen/perjudican de forma diferente las predicciones de distintas longitudes.

El comportamiento también difiere de ELPD: allí no se observaron truncamientos graves de MDiff4STR-Small en la revisión visual; **sí aparecen** en UC3M-LP. Esto no prueba por sí solo una pérdida de generalización, pero sí una diferencia descriptiva relevante entre ambos conjuntos de test.

## 6. Recomendaciones para cerrar la evaluación

1. Definir una **métrica parcial validada** que permita omitir opcionalmente las posiciones anonimizadas y exija conservar los caracteres conocidos en orden, sin premiar inserciones arbitrarias.
2. Mantener **separadas** la evaluación sobre UC3M-LP y la evaluación completa de EM/CER sobre ELPD.
3. Agrupar los fallos visuales por matrícula de una/dos líneas, calidad e inclinación, comparando las mismas imágenes entre los cuatro modelos.
4. Reservar una evaluación de **generalización sin máscaras** para un conjunto externo con etiquetas completas.

## 7. Material revisado y reproducibilidad

- `all_predictions_review.csv`: 491 filas.
- `errors.csv`: 404 discrepancias frente al texto visible.
- `error_analysis.md`: informe generado automáticamente.
- `contact_sheet_errors_01.jpg` a `contact_sheet_errors_05.jpg`.
- `contact_sheet_truncations.jpg`.
- `contact_sheet_high_confidence_errors.jpg`.
- `contact_sheet_mask_ambiguity.jpg`.
- `contact_sheet_low_confidence_visible_exact.jpg`.
- `MDiff4STR_Small_UC3M_casos_prioritarios.csv` (archivo complementario): **87 predicciones** seleccionadas mediante reglas conservadoras; no significa que las 87 se hayan revisado visualmente una por una.

**Modelo/configuración de referencia:** MDiff4STR-Small pretrained, `sampler_step=3`, `sample_k=3`, modo semiautorregresivo, `seed=42`; NVIDIA A100-PCIE-40GB, batch 8. Checkpoint SHA-256 `ebc96f11e5739eae98491f7c8c140157ed12b35dafa220511ef9d399cc20a937`; manifest SHA-256 `97a6e041a996f2865796780f44b41ef4318b1c25066685e8293deb5df1939194`.
