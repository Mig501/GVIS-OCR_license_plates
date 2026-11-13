# UC3M-LP — Revisión visual del baseline SVTRv2-Small

**Proyecto:** OCR de matrículas — Prácticas GVIS  
**Modelo:** SVTRv2-Small (OpenOCR; pretrained, zero-shot, sin fine-tuning)  
**Partición:** UC3M-LP `test` — 491 recortes  
**Fecha de revisión:** 13 de noviembre de 2026  
**Material examinado:** ZIP `error_analysis(2).zip` con `all_predictions_review.csv` (491 registros), `errors.csv` (341 discrepancias), `error_analysis.md` y nueve hojas de contacto (`contact_sheet_*.jpg`).

## 1. Alcance y criterio de revisión

Se han inspeccionado las cinco hojas generales (60 casos mostrados, ordenados por distancia respecto al texto visible), más las cuatro hojas temáticas de truncamientos, confianza alta, ambigüedad de las máscaras y coincidencias visibles de confianza relativamente baja. Hay imágenes repetidas entre hojas: **no es una inspección manual exhaustiva de las 491 matrículas**.

El *ground truth* parcial incorpora `*` en posiciones anonimizadas. El contenido real detrás de cada máscara es desconocido. Se distingue entre:

- **Error confirmado:** falta o se altera algún carácter **conocido/visible**, o la predicción tiene una longitud superior a la de la cadena original con máscaras.
- **Discrepancia ambigua:** el modelo añade caracteres frente a `ground_truth_visible`, pero pueden corresponder a posiciones anonimizadas. No se contabilizan automáticamente como errores reales.
- **Posible factor visual:** desenfoque, oscuridad, iluminación excesiva, perspectiva, formato de dos líneas o disposición gráfica, observable en la imagen; no constituye por sí solo demostración causal del error.

## 2. Resultados cuantitativos

| Indicador | Resultado |
|---|---:|
| Imágenes de test evaluadas | **491/491** |
| EM parcial estricto (longitud íntegra y posiciones conocidas) | **1,43 % (7/491)** |
| Tasa de edición con máscara (proxy, **no CER completo**) | **40,32 % (991/2.458)** |
| Predicciones de longitud total correcta | **60/491** |
| Coincidencia exacta frente al texto visible sin `*` (diagnóstico) | **30,55 % (150/491)** |
| Tasa de edición frente al texto visible (diagnóstico) | **20,95 % (515/2.458)** |
| Predicciones diferentes del texto visible | **341/491** |
| Predicciones más largas que el texto visible | **317/491** |
| Predicciones más cortas que el texto visible | **13/491** |
| Texto visible presente exactamente como sufijo de una predicción más larga | **178/491** |
| Predicciones vacías | **1/491** |
| Predicciones que superan la longitud total del GT parcial | **18/491** |
| Predicciones que **no** contienen todos los caracteres conocidos como subsecuencia en el orden correcto | **50/491** |

Las dos últimas comprobaciones se calcularon a partir de `all_predictions_review.csv`. Son diagnósticos adicionales que **no dependen de adivinar las posiciones ocultas**: una matrícula completa válida no puede superar la longitud total anotada y debe conservar los caracteres conocidos en su orden. Que una salida pase estos filtros no demuestra, por sí solo, que sea correcta.

**Clasificación de las 341 discrepancias frente al texto visible (Levenshtein, solo descriptiva):**

| Categoría | Casos |
|---|---:|
| Solo inserciones | 291 |
| Solo sustituciones | 10 |
| Solo eliminaciones | 3 |
| Mezclas de operaciones | 37 |
| **Total** | **341** |

Las operaciones acumuladas de una alineación óptima son 425 inserciones, 53 sustituciones y 37 eliminaciones. Dado que las posiciones `*` no se ven, **no deben presentarse como 425 inserciones reales de OCR**.

## 3. Errores y comportamientos observados visualmente

### 3.1. Truncamientos severos y predicción vacía — confirmados

Aparecen salidas de uno o ningún carácter cuando permanecen visibles varios caracteres en la imagen. Se observan tanto matrículas de una fila como de dos filas.

| Identificador de imagen | GT parcial | Predicción | Evidencia / interpretación |
|---|---|---|---|
| `test__00606__lp01.png` | `624*GX*` | *(vacía)* | Se distinguen `624` y `GX`, pero no hay salida; confianza 0,213 |
| `test__00207__lp00.png` | `*166DW*` | `G` | Solo un carácter frente a `166DW` visible; placa muy oscura; confianza 0,081 |
| `test__00818__lp00.png` | `7*03C*M` | `D` | Placa de dos líneas; se reconocen visualmente caracteres en ambas filas, predicción de un único carácter; confianza 0,137 |
| `test__01454__lp00.png` | `*652F*M` | `L` | Omite `652F` y `M` visibles; confianza 0,132 |
| `test__00479__lp00.png` | `737**YN` | `N` | No reproduce los números iniciales ni la `Y`; imagen oscura; confianza 0,097 |

### 3.2. Omisión de una línea o de letras finales — confirmada

En recortes de dos líneas es frecuente que el modelo se quede con los números de la parte superior y pierda letras de la inferior. La muestra permite identificar un **patrón visual**, pero no cuantificar su prevalencia en todo el dataset sin clasificar todas las placas por formato.

| Identificador de imagen | GT parcial | Predicción | Observación |
|---|---|---|---|
| `test__00771__lp00.png` | `336*LS*` | `43361` | Se omiten las letras visibles `LS` de la línea inferior |
| `test__00798__lp00.png` | `211**NG` | `D211` | Se omiten las letras inferiores `NG` |
| `test__00817__lp01.png` | `36*7KW*` | `0367` | Omite `KW`, situadas en la fila inferior |
| `test__00358__lp00.png` | `999**TW` | `099991` | No reproduce `TW` de la fila inferior |
| `test__00979__lp00.png` | `5*30*FF` | `1830` | Pierde `FF`; la imagen está muy sobreexpuesta |

### 3.3. Inserciones y prefijos — en su mayor parte ambiguos por anonimización

Hay 317 predicciones más largas que el texto visible y, dentro de estas, 178 terminan exactamente en todos los caracteres visibles, con uno o varios símbolos delante. Los prefijos simples más frecuentes dentro de ese grupo son `8` (43 casos), `1` (33) y `4` (30). **No se pueden atribuir automáticamente a errores**, pues los caracteres ocultos pueden haberse omitido en la etiqueta visible.

Ejemplos de discrepancia **no concluyente**:

- `test__00000__lp00.png`: `597*LK* → 4597LK`. La `4` inicial podría ser una inserción, pero no es posible descartarla formalmente con la GT parcial.
- `test__00015__lp00.png`: `4*73B*W → 15473BW`. Se conservan los caracteres conocidos en orden, pero el ajuste a las máscaras y la longitud no permite declararla correcta.

Casos donde **sí** existe al menos un problema objetivo:

- `test__00326__lp00.png`: `077*KS* → 4077KSKS`. Hay repetición de `KS` y **8 caracteres predichos** frente a 7 posiciones anotadas: longitud excesiva confirmada (la imagen es oscura).
- `test__01515__lp00.png`: `543*FZ* → 185431221`. Hay **9 caracteres predichos** frente a 7 posiciones; tampoco se preservan las letras visibles `FZ`.
- `test__01667__lp01.png`: `146*L*J → 10146LLDJ`. Salida de 9 caracteres para 7 posiciones anotadas. La imagen muestra cierta perspectiva.

**No se identifica, en las hojas revisadas, una repetición sostenida de largas secuencias comparable a los ejemplos extremos de SVTRv2 en ELPD.** Hay duplicaciones puntuales y salidas excesivamente largas, pero no conviene extrapolarlas al conjunto de test.

### 3.4. Confusiones entre caracteres visualmente semejantes — indicios, no patrón confirmado

Algunos casos pueden ser compatibles con las confusiones habituales `0/O`, `1/I` y `8/B`; por ejemplo, `test__00555__lp01.png` (`0*05K*H → LOOSKH`) o `test__00374__lp00.png` (`0*08*DR → LOO8DR`). Sin embargo, las máscaras y las posibles inserciones impiden atribuir con seguridad cada carácter de salida a una posición visible concreta.

Por ello, **no se ha confirmado una frecuencia específica de estas confusiones** para SVTRv2-Small en UC3M-LP.

### 3.5. Calidad visual y distribución de la matrícula — factores asociados

En los casos revisados aparecen ejemplos con:

- **Oscuridad y bajo contraste** (`test__00207__lp00.png`, `test__00479__lp00.png`).
- **Sobreexposición y pérdida de detalle** (`test__00979__lp00.png`).
- **Dos líneas de caracteres** (`test__00818__lp00.png`, `test__00771__lp00.png`, `test__00798__lp00.png`).
- **Perspectiva pronunciada o recortes pequeños** (`test__01684__lp00.png`, `test__01803__lp00.png`).

No hay evidencia suficiente en las hojas para concluir que los escudos nacionales causen errores concretos; tampoco puede afirmarse, con estos ejemplos, que la banda lateral `E` se reconozca sistemáticamente como carácter.

### 3.6. Confianza elevada en fallos reales

La hoja de confianza elevada contiene numerosas **discrepancias ambiguas** porque comparan contra el GT visible. No deben presentarse todas como predicciones erróneas.

Sí se identifica un ejemplo claro de omisión conocida:

- `test__01307__lp01.png`: `56*6*CR → 56CR`, **confianza ≈ 0,913**. Falta el `6` visible de la cuarta posición de GT y la predicción no puede contenerlo oculto en otro lugar de la secuencia.

Por tanto, la confianza no garantiza que el modelo haya recuperado todos los caracteres visibles. No se interpreta como probabilidad calibrada de exactitud.

## 4. Comparación con baselines ya revisados

| Modelo | EM parcial estricto ↑ | Error con máscara (proxy) ↓ | Coincidencia exacta con texto visible ↑ |
|---|---:|---:|---:|
| **PARSeq** | 1,63 % | 39,83 % | **72,91 %** |
| **SVTRv2-Tiny** | **2,44 %** | 43,25 % | 11,20 % |
| **SVTRv2-Small** | 1,43 % | **40,32 %** | 30,55 % |

SVTRv2-Small mejora claramente a Tiny según la coincidencia exacta del texto visible y el error con máscara, pero ambos mantienen truncamientos y pérdida de caracteres visibles. PARSeq continúa siendo superior en coincidencia diagnóstica del texto visible. El EM parcial estricto no refleja adecuadamente estas diferencias: exige longitud completa y posiciones conocidas correctas, aunque existan dos posiciones anonimizadas.

**Importante:** las métricas parciales de UC3M-LP no son directamente comparables con EM/CER completos de ELPD. La coincidencia exacta frente al texto visible también es una métrica auxiliar con sesgos propios.

## 5. Notas breves para interpretar las hojas (tutor)

- **Truncamiento extremo y salida vacía:** el modelo devuelve uno o ningún carácter aun cuando se distinguen varios en la placa.
- **Omisión en matrículas de dos líneas:** puede reconocer la parte superior y perder letras visibles de la inferior.
- **Inserción o repetición de caracteres:** se dan predicciones de longitud excesiva; solo es un error seguro si exceden la longitud original o se altera texto conocido.
- **Anonimización (`*`):** muchas inserciones frente al GT visible pueden corresponder a caracteres ocultos y **no son errores confirmados**.
- **Imágenes difíciles:** oscuridad, brillo excesivo y perspectiva aparecen asociados a varios fallos, sin causalidad probada.
- **Confianza alta no equivale a acierto:** pueden faltar caracteres visibles con puntuaciones superiores a 0,9.

## 6. Conclusión y siguientes pasos

**SVTRv2-Small mejora significativamente a SVTRv2-Tiny en el diagnóstico del texto visible**, pero conserva truncamientos severos, omisiones en matrículas de dos líneas y algunas predicciones más largas que la propia matrícula. La gran proporción de discrepancias clasificadas como inserciones está **amplificada por las posiciones anonimizadas**; por tanto, no debe presentarse sin matices como una tasa de errores de inserción real.

Para completar el análisis comparativo con el mismo nivel de revisión visual falta examinar **MDiff4STR-Small**. Después conviene validar una métrica de alineación que permita que cada posición `*` esté presente u omitida sin penalizarla como un fallo conocido y examinar específicamente las placas de dos líneas, usando un etiquetado de formato verificable.

---

**Trazabilidad:** los valores de métricas y categorías proceden de `error_analysis.md` y `all_predictions_review.csv`; las descripciones de errores visuales se limitan a las nueve hojas de contacto entregadas. El análisis no modifica ni los PNG ni las anotaciones originales.
