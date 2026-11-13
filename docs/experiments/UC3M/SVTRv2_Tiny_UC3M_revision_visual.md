# UC3M-LP — Revisión visual de errores del baseline SVTRv2-Tiny

**Modelo:** SVTRv2-Tiny · **Evaluación:** zero-shot, sin fine-tuning · **Partición:** test, 491 recortes · **Fuente:** `error_analysis(1).zip` (`all_predictions_review.csv`, `errors.csv`, `error_analysis.md` y nueve hojas de contacto).

## 1. Protocolo y precauciones

- El *ground truth* (GT) contiene caracteres anonimizados con `*`. Los rectángulos ocultan el carácter verdadero y **no es posible verificar su identidad**.
- Se revisaron **nueve hojas de contacto**: cinco hojas generales (60 ejemplos) y cuatro temáticas (48 ejemplos, algunos solapados). Es una **revisión visual selectiva**, no una inspección exhaustiva de los 491 recortes ni de las 436 discrepancias.
- Las inserciones, sustituciones y eliminaciones indicadas por el script se calculan frente al **texto visible** (`ground_truth_visible`, obtenido eliminando los `*`). Son **operaciones de alineamiento**, no necesariamente errores reales del OCR: el modelo puede generar caracteres en posiciones cuya etiqueta fue ocultada.
- Una predicción **más larga que el GT completo** constituye una discrepancia real de longitud, pero no permite deducir qué carácter concreto sobra. Una predicción más larga **solo que el GT visible** no confirma una inserción.
- Los ejemplos y las confianzas proceden del CSV incluido en el ZIP; las observaciones de legibilidad y disposición se apoyan en las miniaturas de las hojas de contacto.

## 2. Resultados cuantitativos verificados

| Indicador | Resultado |
|---|---:|
| Recortes evaluados | **491** |
| EM parcial estricto (requiere longitud completa y acierto en posiciones conocidas) | **2,44 % (12/491)** |
| Distancia de edición con máscara / caracteres conocidos (*proxy*; **no CER**) | **43,25 % (1.063/2.458)** |
| Predicciones con longitud del GT original | **107/491** |
| Coincidencia exacta con el GT visible (diagnóstico) | **11,20 % (55/491)** |
| Edición respecto al GT visible (diagnóstico) | **30,92 % (760/2.458)** |
| Predicciones distintas del texto visible | **436/491** |
| Más largas / más cortas que el GT visible | **404 / 15** |
| Más largas que el GT completo | **48** |
| GT visible como sufijo de una predicción más larga | **216** |
| Predicciones vacías | **0** |

**Desglose descriptivo de las 436 discrepancias frente al texto visible:** solo inserciones, 367; solo sustituciones, 7; solo eliminaciones, 5; errores mixtos, 57. En los alineamientos se registran **639 inserciones, 71 sustituciones y 50 eliminaciones**, que **no deben citarse como errores de OCR confirmados**.

La mayor parte de las etiquetas contienen siete posiciones; **tres etiquetas contienen ocho**. En las notas de revisión debe comprobarse la longitud correspondiente a cada caso y no suponer siempre que el GT mide siete.

## 3. Hallazgos de la inspección visual

### 3.1. Prefijos de caracteres añadidos: patrón predominante

Se aprecia una tendencia a generar caracteres delante de la secuencia visible. En **216/491 casos**, el texto conocido aparece íntegramente como sufijo de una predicción más larga. Los prefijos más frecuentes dentro de esos 216 casos son `1` (88), `8` (38) y `4` (38); también aparecen prefijos compuestos como `18`.

| Imagen | GT parcial | Predicción | Interpretación |
|---|---|---|---|
| `test__00000__lp00.png` | `597*LK*` | `185970LK7` | **9 caracteres frente a 7 posiciones** y `18` antes del bloque inicial `597`; hay longitud excedida y prefijo espurio. |
| `test__01515__lp00.png` | `543*FZ*` | `18543TFZ1` | **9 frente a 7 posiciones**; presenta el prefijo `18` delante de `543` y otros caracteres adicionales. |
| `test__00186__lp00.png` | `M218*Z*` | `EM2181Z1` | Aparece una `E` **antes de la `M` visible**; en la miniatura se distingue una `E` en la banda lateral de la placa. Es un indicio de lectura del distintivo lateral, no una causa comprobada experimentalmente. |
| `test__01234__lp00.png` | `142**KY` | `181421KY` | Prefijo `18`; el GT visible aparece al final, pero la correspondencia con las posiciones tapadas sigue siendo incierta. |

**Precaución:** no se puede afirmar que los **216** casos sean **216 inserciones reales**: los caracteres generados también pueden estar desplazados respecto a las posiciones anonimizadas. Los casos de longitud superior al GT completo y los prefijos que preceden a los primeros caracteres conocidos ofrecen evidencias más sólidas.

### 3.2. Truncamientos graves en matrículas de una línea

La predicción puede terminar después de uno, tres o cuatro caracteres y omitir grupos de letras o números claramente visibles en el recorte.

| Imagen | GT parcial | Predicción | Evidencia visual |
|---|---|---|---|
| `test__00393__lp00.png` | `R*366BD*` | `R` | Solo devuelve `R`, perdiendo `366BD` (confianza **0,270**; imagen degradada). |
| `test__00311__lp00.png` | `*767*BW` | `B` | Apenas conserva un carácter de los cinco conocidos (confianza **0,309**). |
| `test__01206__lp00.png` | `774**JC` | `18774` | Reconoce `774`, pero **omite `JC`**, visible a la derecha, con confianza **0,875**. |
| `test__01563__lp00.png` | `718**NT` | `718` | Omite **`NT`**, aunque ambas letras se distinguen; confianza **0,887**. |
| `test__00251__lp00.png` | `914**CP` | `4914` | Omite **`CP`**, claramente presente; confianza **0,844**. |
| `test__00353__lp00.png` | `255*CM*` | `4255` | Omite **`CM`**; confianza **0,841**. |

Estos casos **sí evidencian pérdida de caracteres conocidos**, independientemente del contenido de las máscaras.

### 3.3. Omisión de la fila inferior en matrículas de dos líneas

En varios recortes con texto distribuido en dos filas, el modelo predice sobre todo los dígitos de la fila superior y **omite letras de la inferior**.

| Imagen | GT parcial | Predicción | Observación visual |
|---|---|---|---|
| `test__00376__lp00.png` | `01*6H*J` | `1016` | Matrícula de dos filas; las letras `H` y `J` inferiores quedan fuera de la predicción. |
| `test__00771__lp00.png` | `336*LS*` | `43361` | Matrícula cuadrada; omite `LS` en la segunda línea. |
| `test__00818__lp00.png` | `7*03C*M` | `47033` | Reconoce parte de la fila superior, pero omite `C` y `M` en la inferior. |
| `test__00358__lp00.png` | `999**TW` | `89991` | Se distingue `TW` en la línea inferior, pero no aparece en la salida. |
| `test__00798__lp00.png` | `211**NG` | `B2111` | Omite `NG` de la fila inferior. |

La **disposición en dos líneas** coincide con varios fallos graves, pero también hay truncamientos en placas de una sola línea; por tanto, **no se puede atribuir causalidad exclusiva** al formato sin evaluaciones controladas por categoría.

### 3.4. Repeticiones aparentes y efecto de la anonimización

No se observan en las hojas revisadas repeticiones extensas de subcadenas comparables a algunos casos de SVTRv2-Tiny en ELPD, pero sí **duplicaciones aparentes de dígitos y letras**, a menudo próximas a los rectángulos de anonimización.

| Imagen | GT parcial | Predicción | ¿Error confirmado? |
|---|---|---|---|
| `test__00188__lp00.png` | `8*14G*Y` | `8814GGY` | **No.** Longitud completa correcta y todas las posiciones conocidas coinciden (`Dmask=0`); los caracteres adicionales ocupan las dos posiciones ocultas. |
| `test__01220__lp00.png` | `61*1JZ*` | `16111JZ1` | **Al menos una discrepancia de longitud:** ocho caracteres frente a siete, con varios `1` consecutivos; no puede asignarse cada `1` a una posición verdadera. |
| `test__01667__lp01.png` | `146*L*J` | `11146TLLJ` | **9 frente a 7 posiciones**; prefijo y repetición aparente, pero sigue sin conocerse el contenido de las máscaras. |

El ejemplo `8*14G*Y → 8814GGY` es especialmente útil para explicar al tutor **por qué contar duplicaciones respecto al GT visible puede producir falsos positivos**.

### 3.5. Confusiones visuales, distintivos y calidad de imagen

- En algunos recortes se aprecian **confusiones posibles** entre formas similares, por ejemplo `0/O`, `1/I` y `5/S`. En `test__01867__lp00.png` (`*101FC* → PLIOTFCO`) la predicción difiere en varios puntos del texto conocido, pero la imagen está muy borrosa y el alineamiento exacto no es inequívoco. **No debe asignarse automáticamente cada sustitución a una letra concreta.**
- `test__00186__lp00.png` (`M218*Z* → EM2181Z1`) ofrece un **indicio visual de lectura de la `E` de la banda lateral** como parte del texto; no constituye una demostración general del mecanismo de error.
- Se observan errores en imágenes **muy oscuras** (`test__00393__lp01.png`), **borrosas o de bajo contraste** (`test__01867__lp00.png`) y **oblicuas** (`test__01651__lp01.png`). Se trata de asociaciones visuales, no de efectos causales cuantificados.
- No hay evidencia suficiente en este material para afirmar que el modelo **confunde escudos nacionales con caracteres** en UC3M-LP. Esta explicación de ELPD **no debe trasladarse sin verificación**.

### 3.6. Confianza elevada con pérdidas de caracteres visibles

La hoja temática de alta confianza contiene numerosos casos **ambiguos** en los que solo se añade un carácter que podría corresponder a una máscara. Conviene emplear los ejemplos siguientes, donde **sí se omiten caracteres conocidos**:

| Imagen | GT parcial | Predicción | Confianza | Caracteres conocidos omitidos |
|---|---|---|---:|---|
| `test__01563__lp00.png` | `718**NT` | `718` | **0,887** | `NT` |
| `test__01206__lp00.png` | `774**JC` | `18774` | **0,875** | `JC` |
| `test__00180__lp00.png` | `767**PY` | `8767` | **0,870** | `PY` |
| `test__00251__lp00.png` | `914**CP` | `4914` | **0,844** | `CP` |
| `test__00353__lp00.png` | `255*CM*` | `4255` | **0,841** | `CM` |

Una confianza alta es una **puntuación del modelo**, no una probabilidad calibrada de reconocimiento correcto; **no basta como único criterio de aceptación**.

## 4. Comparación provisional con PARSeq (mismo test)

| Indicador | PARSeq | SVTRv2-Tiny |
|---|---:|---:|
| EM parcial estricto ↑ | 1,63 % (8/491) | **2,44 % (12/491)** |
| Error con máscara (*proxy*) ↓ | **39,83 %** | 43,25 % |
| Coincidencia exacta con GT visible ↑ | **72,91 % (358/491)** | 11,20 % (55/491) |
| Edición frente a GT visible ↓ | **8,62 %** | 30,92 % |
| Más largas que el GT visible | 95 | 404 |
| Predicciones vacías | 0 | 0 |

PARSeq transcribe con mucha mayor frecuencia exactamente la cadena de caracteres visibles; Tiny produce más prefijos y más predicciones largas, y en la inspección aparecen omisiones importantes de caracteres conocidos. **No obstante, ambas métricas auxiliares penalizan de forma distinta los caracteres de posiciones anonimizadas** y no representan una tasa de matrícula completa correctamente reconocida.

## 5. Conclusión provisional

SVTRv2-Tiny muestra **numerosos prefijos y caracteres adicionales**, **truncamientos severos** y **omisión de segmentos visibles**, particularmente en algunas matrículas de dos líneas. Su tasa de coincidencia exacta con texto visible (11,20 %) es sensiblemente inferior a la de PARSeq (72,91 %), pero **ninguna de esas cifras permite juzgar sin reservas la matrícula completa** porque el GT incluye posiciones ocultas. Las repeticiones cercanas a `*` deben tratarse como **ambiguas**, salvo que se pueda demostrar un exceso de longitud o una incompatibilidad con posiciones conocidas.

La revisión visual inicial de Tiny queda documentada. El siguiente paso para la serie es revisar **SVTRv2-Small** con el mismo protocolo y, antes del ranking final, validar una métrica de alineamiento que permita omitir o rellenar las posiciones anonimizadas sin atribuirles caracteres supuestos.
