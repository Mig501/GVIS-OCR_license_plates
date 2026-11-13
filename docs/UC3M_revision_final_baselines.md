# Revisión final de los baselines OCR sobre UC3M-LP

**Proyecto:** Reconocimiento OCR de matrículas — Prácticas GVIS  
**Dataset:** UC3M-LP (partición `test`)  
**Modelos evaluados:** PARSeq, SVTRv2-Tiny, SVTRv2-Small y MDiff4STR-Small  
**Fase:** Evaluación inicial *zero-shot*, sin fine-tuning  
**Estado:** Análisis cuantitativo documentado; observaciones visuales de errores consolidadas a partir de cuatro revisiones selectivas.

## 1. Objetivo y protocolo de evaluación

Se comparan cuatro modelos de reconocimiento de texto sobre los mismos **491 recortes de test de UC3M-LP**, con el objetivo de establecer una referencia previa al ajuste fino e identificar dificultades de lectura comunes y específicas de cada arquitectura.

- **Conjunto evaluado:** 491 imágenes; las etiquetas contienen caracteres conocidos y posiciones anonimizadas mediante `*`. La mayoría de las etiquetas tienen siete posiciones y tres tienen ocho.
- **Condición experimental:** inferencia con modelos preentrenados, sin fine-tuning sobre UC3M-LP.
- **Normalización:** conversión de las predicciones a mayúsculas y conservación de caracteres ASCII `A–Z` y `0–9`.
- **EM parcial estricto:** proporción de predicciones con la misma longitud que el GT parcial y coincidencia de **todos los caracteres conocidos en sus posiciones anotadas**. No verifica el contenido de las posiciones ocultas.
- **Tasa de edición con máscara (*proxy*):** suma de distancias de Levenshtein modificadas dividida entre el número de caracteres conocidos (2.458). Sustituir un `*` por cualquier carácter no penaliza, pero insertar o eliminar caracteres cuesta 1, incluso en posiciones anonimizadas. **No equivale al CER convencional** y penaliza predicciones que omiten posiciones tapadas.
- **Coincidencia exacta con el texto visible (auxiliar):** porcentaje de predicciones que coinciden con el GT una vez eliminados los `*`. Su distancia de edición asociada también se calcula respecto a esa cadena reducida; **no mide el reconocimiento completo de la matrícula** y penaliza caracteres que podrían corresponder a posiciones ocultas.

Las revisiones visuales de cada modelo se basan en **nueve hojas de contacto** (cinco generales y cuatro temáticas), con posibles imágenes repetidas. **No representan una inspección individual exhaustiva de las 491 imágenes.** Los errores descritos se distinguen de las discrepancias producidas por el procedimiento de anonimización. Una asociación con el formato de dos filas, la iluminación o los distintivos **no demuestra por sí sola causalidad**.

## 2. Resultados globales

| Modelo | EM parcial estricto ↑ | Error con máscara (*proxy*) ↓ | Coincidencia exacta visible (aux.) ↑ | Edición visible (aux.) ↓ | Discrepancias con texto visible | Parámetros |
|---|---:|---:|---:|---:|---:|---:|
| **PARSeq** | 1,63 % (8/491) | 39,83 % (979/2.458) | **72,91 % (358/491)** | **8,62 % (212/2.458)** | **133** | 23,83 M |
| **MDiff4STR-Small** | **4,28 % (21/491)** | **38,00 % (934/2.458)** | 17,72 % (87/491) | 24,00 % (590/2.458) | 404 | 18,89 M |
| **SVTRv2-Small** | 1,43 % (7/491) | 40,32 % (991/2.458) | 30,55 % (150/491) | 20,95 % (515/2.458) | 341 | 12,52 M |
| **SVTRv2-Tiny** | 2,44 % (12/491) | 43,25 % (1.063/2.458) | 11,20 % (55/491) | 30,92 % (760/2.458) | 436 | 5,71 M |

**Lectura principal:** **PARSeq** obtiene la mayor coincidencia con la cadena de caracteres visibles, mientras **MDiff4STR-Small** encabeza el EM parcial estricto y la tasa de edición con máscara. SVTRv2-Small mejora frente a Tiny en las dos métricas auxiliares del texto visible y en la tasa de edición con máscara, aunque su EM parcial estricto es ligeramente inferior. **Estas diferencias no permiten establecer una clasificación definitiva de reconocimiento de matrículas completas**, porque el contenido real de los `*` es desconocido y las métricas favorecen de manera distinta las longitudes de salida.

La cifra de «discrepancias con texto visible» procede de la comparación automática con la cadena sin `*`: **no equivale al número de errores de OCR confirmados**. Los tiempos de inferencia disponibles no se presentan como un benchmark homogéneo: por ejemplo, los 2,629 ms/imagen registrados para MDiff4STR-Small corresponden al *forward* de GPU con batch 8, no a la latencia de extremo a extremo.

## 3. Errores comunes detectados

### 3.1. Truncamientos y omisión de caracteres visibles

Los cuatro modelos presentan ejemplos en los que se pierden letras o números **conocidos**, por lo que el error no depende de adivinar las posiciones anonimizadas. Destacan las matrículas con **dos filas**, en las que algunos reconocedores devuelven solamente caracteres de la parte superior y omiten los de la inferior. También se observaron truncamientos en matrículas de **una sola línea**, de modo que el formato no es una explicación única demostrada.

Un ejemplo compartido es `test__00771__lp00.png` (`336*LS*`): PARSeq devuelve `3361`, SVTRv2-Tiny y SVTRv2-Small devuelven `43361`, y MDiff4STR-Small devuelve `0`. En todos los casos se pierden caracteres conocidos como `LS`. Las causas concretas requieren evaluación por formato y comprobación sobre las imágenes originales.

### 3.2. Prefijos, caracteres adicionales y repeticiones aparentes

Aparecen predicciones más largas que el GT **visible**, a menudo con prefijos como `1`, `8`, `4` o `18`. **No deben considerarse automáticamente inserciones**, porque parte de esos símbolos podría corresponder a posiciones anonimizadas. Son discrepancias verificables cuando la salida **supera la longitud total del GT parcial** o contradice las posiciones conocidas.

Se observan también repeticiones de números o letras. Deben distinguirse las duplicaciones *aparentes* —posiblemente situadas en posiciones ocultas— de los casos con **longitud total excedida**, que sí suponen una incompatibilidad objetiva con la anotación. A diferencia de ELPD, en las hojas de UC3M-LP no se documentó un patrón general de repeticiones largas y sostenidas de subcadenas.

### 3.3. Calidad visual, disposición y confianza

Entre los casos problemáticos hay recortes **oscuros**, **desenfocados**, **sobreexpuestos**, **inclinados** o con texto distribuido en dos filas. Son factores observados visualmente, pero sus efectos no se han aislado mediante un experimento controlado. Existen asimismo truncamientos u omisiones de caracteres conocidos con **confianza elevada**: la puntuación del modelo no es una probabilidad calibrada de transcripción correcta.

Se aprecian **posibles** confusiones entre caracteres semejantes (`0/O`, `1/I`, `8/B`), aunque las máscaras y los alineamientos ambiguos impiden cuantificar una frecuencia fiable con las hojas disponibles. Tampoco hay evidencia para atribuir de forma general los fallos de UC3M-LP a escudos nacionales; en SVTRv2-Tiny sí aparece un indicio puntual de lectura de la `E` de una banda lateral.

## 4. Revisión por modelo

### 4.1. PARSeq

**Resultado:** EM parcial estricto del **1,63 %**; error con máscara del **39,83 %**; coincidencia exacta con el texto visible del **72,91 %** (358/491).

**Errores detectados:**

- Presenta **truncamientos confirmados en matrículas de dos filas**, como `336*LS* → 3361` y `36*7KW* → 367`, donde se omiten letras conocidas de la línea inferior.
- También se observan **omisiones en matrículas de una fila**, como `694**XM → 694` y `949**TS → 949`.
- Hay **repeticiones o salidas excesivamente largas** en algunos casos: `505**WW → 5050WWWW` y `686**MM → 686PMMMM` producen ocho caracteres frente a siete posiciones anotadas. En cambio, `733**JW → 73333JW` no puede declararse errónea solo por la duplicación aparente, ya que tiene longitud correcta y respeta las posiciones conocidas.
- Se detectan **errores con confianza alta**, por ejemplo `2*28L*L → 28LL` (0,940), en el que desaparece un `2` conocido.

**Interpretación:** ofrece la **mejor coincidencia con los caracteres visibles condensados** y el menor error de edición en esa comparación auxiliar. Sin embargo, esto no verifica matrículas completas: parte del bajo EM parcial se explica por predicciones que no reproducen las posiciones ocultas. Los truncamientos y las omisiones de caracteres conocidos son fallos reales observados, no simples efectos de la máscara.

### 4.2. SVTRv2-Tiny

**Resultado:** EM parcial estricto del **2,44 %**; error con máscara del **43,25 %**; coincidencia exacta con texto visible del **11,20 %** (55/491).

**Errores detectados:**

- Destacan los **prefijos** y predicciones de longitud excedida. Por ejemplo, `597*LK* → 185970LK7` tiene nueve caracteres frente a siete posiciones; en `M218*Z* → EM2181Z1` se observa una `E` delante de la `M` conocida, coincidente visualmente con la banda lateral, sin que ello demuestre causalidad.
- Presenta **truncamientos graves en matrículas de una línea**, como `R*366BD* → R`, `774**JC → 18774` y `718**NT → 718`.
- También **omite caracteres de la segunda fila**, por ejemplo `01*6H*J → 1016` y `7*03C*M → 47033`.
- Hay **omisiones de letras visibles con confianza elevada**, como `718**NT → 718` (0,887) o `774**JC → 18774` (0,875).
- Algunas **duplicaciones son ambiguas**: `8*14G*Y → 8814GGY` respeta la longitud anotada y todas las posiciones conocidas, por lo que no se puede clasificar como error confirmado.

**Interpretación:** es el modelo con **mayor tasa de edición con máscara** y menor coincidencia visible de los cuatro. La revisión destaca prefijos, salidas excesivamente largas y omisiones de segmentos conocidos. No se deben equiparar sus 436 discrepancias frente al texto visible con 436 errores OCR reales.

### 4.3. SVTRv2-Small

**Resultado:** EM parcial estricto del **1,43 %**; error con máscara del **40,32 %**; coincidencia exacta con texto visible del **30,55 %** (150/491).

**Errores detectados:**

- Hay **truncamientos severos y una predicción vacía**, como `624*GX* → [vacía]`. En `*166DW* → G` y `7*03C*M → D` prácticamente desaparecen todos los caracteres conocidos.
- Se repite la **omisión de letras inferiores en matrículas de dos filas**, por ejemplo `336*LS* → 43361`, `211**NG → D211` y `36*7KW* → 0367`.
- Se observan **prefijos adicionales**, muchos de atribución ambigua por las máscaras; algunos casos exceden la longitud completa: `077*KS* → 4077KSKS` y `543*FZ* → 185431221`.
- Hay **errores de caracteres conocidos con confianza alta**, como `56*6*CR → 56CR` (aprox. 0,913), donde falta un `6` conocido.

**Interpretación:** mejora a Tiny en el diagnóstico del texto visible y reduce la tasa de edición con máscara, pero conserva truncamientos importantes, omisiones de segunda línea y algunas salidas excesivamente largas. **No se observa en esta revisión una repetición extensa de secuencias como las de los casos extremos de ELPD**. Su único caso de predicción vacía registrado exige comprobación específica del preprocesamiento y del decodificador.

### 4.4. MDiff4STR-Small

**Resultado:** EM parcial estricto del **4,28 %**; error con máscara del **38,00 %**; coincidencia exacta con texto visible del **17,72 %** (87/491).

**Errores detectados:**

- Se identifican **truncamientos graves**, especialmente en matrículas de **dos líneas**: `336*LS* → 0`, `7*03C*M → L` y `36*7KW* → 7`. También hay omisiones en imágenes de una línea, oscuras o degradadas.
- Hay **pérdidas de caracteres conocidos** como `MA*117*S → MA11S` (confianza 0,909) y `80*8*YF → 88`.
- Se observan **prefijos y duplicaciones**, incluyendo predicciones que exceden la longitud total: `5*30*FF → TB30FFFF` (8 frente a 7), `92*2FP* → 89202FPII` (9 frente a 7) y `677**XV → 16777XXV` (9 frente a 7).
- Aparecen **posibles confusiones entre `0` y `O`**, por ejemplo `98*0*CY → 898OCY`, sin que las hojas permitan cuantificar su incidencia.
- La revisión no registra salidas vacías; sí **tres predicciones de un solo carácter** y casos con confianza alta a pesar de errores comprobables.

**Interpretación:** obtiene el **mejor EM parcial estricto y la menor tasa de edición con máscara**, pero no la mejor coincidencia con el texto visible. Sus **truncamientos graves en UC3M-LP contrastan** con la revisión visual de ELPD, donde no se habían identificado casos comparables. Esto describe una diferencia entre los conjuntos revisados, no demuestra por sí mismo una causa ni una pérdida cuantificada de generalización.

## 5. Comparación de patrones

| Patrón | PARSeq | SVTRv2-Tiny | SVTRv2-Small | MDiff4STR-Small |
|---|---|---|---|---|
| Omisión de caracteres conocidos | **Observada** | **Observada** | **Observada** | **Observada** |
| Truncamiento grave | Presente | **Presente** | **Presente** | **Presente** |
| Omisión de la segunda fila | Observada | Observada | Observada | Observada |
| Prefijos y texto adicional respecto al GT visible | Presente; muchas discrepancias ambiguas | **Muy destacado** | **Destacado** | **Destacado** |
| Longitud superior a la del GT completo | Casos revisados | **48 predicciones** | **18 predicciones** | **11 predicciones** |
| Repeticiones aparentes | Puntuales; algunas ambiguas | Puntuales; algunas ambiguas | Puntuales; algunas ambiguas | Puntuales; algunas exceden longitud |
| Salida vacía | No registrada | No registrada | **1 caso** | No registrada |
| Salidas de un solo carácter | No destacado | Observadas | Observadas | **3 casos** |
| Error con confianza elevada en caracteres conocidos | Confirmado | Confirmado | Confirmado | Confirmado |
| Elementos laterales asociados visualmente a fallos | Sin patrón confirmado | Indicio puntual de la banda `E` | Sin patrón confirmado | Sin patrón dominante confirmado |

*«Presente» u «observada» indica hallazgos en las revisiones disponibles, no una frecuencia medida sobre todos los recortes. «No destacado» no equivale a ausencia. Las cifras de longitud superior al GT completo son incompatibilidades objetivas, mientras que un exceso frente al GT visible puede ser legítimo por los `*`.*

## 6. Conclusiones

1. **La anonimización impide medir EM y CER completos** sobre UC3M-LP. El EM parcial estricto, el error con máscara y la coincidencia con texto visible responden a criterios distintos; no deben combinarse en una clasificación única de precisión real.
2. **PARSeq destaca en la reproducción exacta de la cadena visible**, con 358/491 coincidencias (72,91 %), pero presenta fallos confirmados por omisión, truncamiento y algunas salidas demasiado largas.
3. **MDiff4STR-Small lidera las métricas parciales originales** (4,28 % de EM parcial estricto; 38,00 % de error con máscara), aunque también incurre en omisiones y truncamientos graves, especialmente en ejemplos de dos filas.
4. **SVTRv2-Small mejora a Tiny en los diagnósticos principales del texto visible**, pero ambos muestran prefijos, omisiones de caracteres conocidos y truncamientos. Small presenta una predicción vacía; Tiny acumula más casos con longitud superior al GT completo (48 frente a 18).
5. **La disposición de dos filas aparece repetidamente entre los fallos de los cuatro modelos**; no obstante, también hay truncamientos de una sola línea. La causa e incidencia del efecto del formato requieren una evaluación estratificada.
6. **Los elementos adicionales, las repeticiones y las confusiones visuales deben analizarse con cautela**. Las posiciones `*` pueden hacer que una predicción parezca contener inserciones o duplicaciones aun cuando sea compatible con los caracteres conocidos.
7. **La confianza alta no garantiza una transcripción válida**. Hay ejemplos de todos los modelos que omiten caracteres verificables pese a puntuaciones elevadas.

## 7. Limitaciones y siguientes verificaciones

- **Anotación parcial:** los caracteres ocultos por `*` no disponen de GT verificable. Una coincidencia con texto visible no confirma la matrícula completa, y una discrepancia frente a ese texto no prueba automáticamente un error OCR.
- **Métricas específicas:** validar una medida de alineamiento que permita que cada `*` corresponda a un carácter desconocido generado o a una posición omitida, con restricciones que impidan premiar inserciones arbitrarias; documentar claramente qué puede y qué no puede medir.
- **Cobertura de revisión:** las nueve hojas por modelo son una selección con solapamientos, no una auditoría visual completa de las 491 imágenes. No extrapolar frecuencias de patrones sin clasificar el conjunto entero.
- **Formato y robustez:** separar matrículas de una y dos filas, así como imágenes oscuras, borrosas, sobreexpuestas o inclinadas, y comparar **las mismas imágenes** entre modelos. Verificar a resolución original los casos extremos.
- **Errores objetivamente identificables:** diferenciar omisiones o alteraciones de caracteres conocidos, longitudes superiores al GT completo y salidas vacías de los casos meramente ambiguos por anonimización.
- **Elementos gráficos y causalidad:** estudiar el efecto de bandas laterales, escudos o recortes ajustados mediante pruebas controladas; no trasladar directamente a UC3M-LP las explicaciones propuestas para ELPD.
- **Eficiencia y generalización:** repetir la comparación temporal con el mismo protocolo extremo a extremo y utilizar también un conjunto externo con **etiquetas completas**. Los resultados sobre ELPD y UC3M-LP **no son directamente comparables** en EM/CER.
- **Trazabilidad:** conservar predicciones originales, informes automáticos, hojas de contacto, configuraciones y hashes de checkpoints. Este documento consolida las cuatro revisiones visuales sin corregir retrospectivamente salidas o etiquetas.

---

**Conclusión final:** en la evaluación inicial sobre UC3M-LP, **PARSeq es el modelo que reproduce con mayor frecuencia la secuencia visible conocida**, mientras que **MDiff4STR-Small obtiene los mejores resultados bajo las métricas parciales estrictas disponibles**. Los cuatro modelos presentan omisiones verificables de caracteres y truncamientos, con especial presencia de fallos en matrículas de dos filas; SVTRv2-Tiny destaca por los prefijos y el exceso de longitud. **La anonimización es la principal limitación metodológica para emitir un ranking de reconocimiento completo**, de modo que las conclusiones deben apoyarse en varios diagnósticos, la revisión visual y una evaluación adicional sobre etiquetas sin máscaras.
