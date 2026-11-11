# Revisión final de los baselines OCR sobre ELPD/Kaggle

**Proyecto:** Reconocimiento OCR de matrículas — Prácticas GVIS  
**Dataset:** European License Plates Dataset (ELPD/Kaggle)  
**Modelos evaluados:** PARSeq, SVTRv2-Tiny, SVTRv2-Small y MDiff4STR-Small  
**Fase:** Evaluación inicial *zero-shot*, sin fine-tuning  
**Estado:** Análisis cuantitativo completado; observaciones visuales de errores documentadas.

## 1. Objetivo y protocolo de evaluación

Se comparan cuatro modelos de reconocimiento de texto sobre el mismo subconjunto auditado de ELPD, con el objetivo de establecer una referencia previa al ajuste fino y detectar las dificultades propias de las matrículas europeas.

- **Conjunto evaluado:** 80 recortes de matrícula del test estricto auditado (76 etiquetas verificadas y 4 corregidas manualmente).
- **Condición experimental:** inferencia con checkpoints preentrenados, sin fine-tuning sobre ELPD.
- **Normalización:** conversión a mayúsculas y conservación únicamente de caracteres ASCII `A–Z` y `0–9`, aplicada de forma común antes del cálculo de métricas.
- **Exact Match (EM):** porcentaje de matrículas cuya predicción normalizada coincide exactamente con la referencia.
- **Character Error Rate (CER):** suma de distancias de Levenshtein entre predicciones y referencias, dividida entre el número total de caracteres de referencia (556). Se consideran inserciones, sustituciones y eliminaciones.

Las observaciones sobre países, escudos, identificadores regionales y causas probables proceden de la inspección de las predicciones y las hojas de contacto. **Una asociación visual no demuestra por sí sola causalidad**; los patrones deben contrastarse mediante pruebas controladas de recorte, supresión de elementos gráficos o postprocesado.

## 2. Resultados globales

| Modelo | Aciertos | Errores de matrícula | EM ↑ | CER ↓ | Errores de edición | Parámetros |
|---|---:|---:|---:|---:|---:|---:|
| **PARSeq** | **55/80** | **25** | **68,75 %** | **6,83 %** | **38** | 23,83 M |
| **MDiff4STR-Small** | 37/80 | 43 | 46,25 % | 14,03 % | 78 | 18,89 M |
| **SVTRv2-Small** | 31/80 | 49 | 38,75 % | 19,96 % | 111 | 12,52 M |
| **SVTRv2-Tiny** | 23/80 | 57 | 28,75 % | 21,04 % | 117 | 5,71 M |

**Lectura principal:** PARSeq obtiene la mejor precisión *zero-shot*. MDiff4STR-Small se sitúa en segundo lugar, seguido de SVTRv2-Small y SVTRv2-Tiny. El aumento de tamaño de SVTRv2 mejora el EM, pero no elimina los patrones de inserción o repetición.

Los tiempos de GPU registrados en esta fase son preliminares y **no constituyen todavía un benchmark de latencia estrictamente comparable**, dado que no todos los pipelines contabilizan de forma equivalente preprocesamiento, decodificación y postprocesamiento.

## 3. Errores comunes detectados

### 3.1. Confusión entre caracteres visualmente similares

Aparece el error conocido de reconocimiento entre caracteres de forma parecida, especialmente **`I`/`1`, `O`/`0` y `B`/`8`**. Puede producir sustituciones aun cuando la mayor parte de la matrícula se reconozca correctamente.

### 3.2. Elementos gráficos y texto ajeno a la secuencia principal

Los escudos, distintivos nacionales, bandas laterales e indicativos regionales aparecen frecuentemente junto a errores de inserción. En determinados recortes, algunos modelos parecen incorporar dichos elementos al texto reconocido, por lo que generan más caracteres de los que contiene la referencia.

### 3.3. Inserciones, repeticiones y truncamientos

- **Inserción:** se añaden caracteres inexistentes en el ground truth, al principio, al final o entre caracteres válidos.
- **Repetición:** se duplica una secuencia ya reconocida, a veces varias veces dentro de la misma predicción.
- **Truncamiento:** la salida termina prematuramente y omite parte sustancial de la matrícula.
- **Sustitución:** se reconoce un carácter diferente en una posición concreta, como `B` por `8`.

Estos fenómenos afectan de forma distinta a los modelos: PARSeq presenta sobre todo inserciones; SVTRv2 muestra inserciones, repeticiones y algunos truncamientos graves; MDiff4STR presenta una mezcla más equilibrada de operaciones de edición.

## 4. Revisión por modelo

### 4.1. PARSeq

**Resultado:** 55/80 aciertos; EM del 68,75 % y CER del 6,83 %.

**Errores detectados:**

- Predominan las **inserciones de caracteres adicionales**, especialmente en zonas próximas a **escudos, distintivos y códigos regionales**.
- Se observan confusiones puntuales entre caracteres similares (`I`/`1`, `O`/`0`, `B`/`8`).
- En casos aislados, la salida se **trunca considerablemente**, como en `GB1ECB → GB`.

**Interpretación:** ofrece el reconocimiento más consistente del conjunto, aunque sigue siendo sensible a componentes gráficos de la placa y presenta algunos errores de longitud de salida. El truncamiento de `GB1ECB` se considera un caso puntual, no su comportamiento predominante.

### 4.2. SVTRv2-Tiny

**Resultado:** 23/80 aciertos; EM del 28,75 % y CER del 21,04 %.

**Errores detectados:**

- Predominan las **inserciones al inicio y al final** de la predicción. Se aprecian especialmente en matrículas francesas, con la `F` del distintivo nacional al principio y el indicativo regional al final.
- Se observan **inserciones asociadas visualmente a escudos**, por ejemplo en matrículas suizas.
- Aparecen **repeticiones anómalas de secuencias**, especialmente en algunos ejemplos de matrículas francesas. Caso representativo: `EQ725QJ → E0725QJ250J250J25`.
- También presenta **truncamientos graves con confianza elevada**, como `KA761ET → KA`. En los ejemplos revisados de este tipo se observa un **escudo croata situado tras los caracteres reconocidos**, circunstancia que podría estar relacionada con la interrupción prematura de la salida.

**Interpretación:** es el modelo que más falla en este conjunto. Su comportamiento exige analizar por separado la influencia de los elementos laterales, la longitud máxima de decodificación y la repetición de tokens, sin atribuir todavía una causa concreta.

### 4.3. SVTRv2-Small

**Resultado:** 31/80 aciertos; EM del 38,75 % y CER del 19,96 %.

**Errores detectados:**

- **Persisten numerosas inserciones y repeticiones**, pese a la mejora global respecto a Tiny.
- Algunas repeticiones son especialmente pronunciadas, como `CY726SV → CY726SV726SV726SV726`.
- Se observan **truncamientos relacionados visualmente con escudos u otros elementos internos** de la matrícula.
- En algunas imágenes, la salida es **vacía o prácticamente inexistente**; conviene distinguir explícitamente estos casos de los reconocimientos parciales.

**Interpretación:** el incremento de capacidad mejora el resultado agregado de Tiny en **10 puntos porcentuales de EM**, pero no resuelve el problema de las repeticiones ni garantiza una mejora en todas las imágenes. Los casos de salida vacía requieren una comprobación específica del decodificador y del preprocesamiento.

### 4.4. MDiff4STR-Small

**Resultado:** 37/80 aciertos; EM del 46,25 % y CER del 14,03 %.

**Errores detectados:**

- Presenta **errores más variados**, con inserciones, sustituciones y eliminaciones.
- En la revisión visual efectuada **no se identificaron truncamientos graves del tipo observado en PARSeq o SVTRv2**. Esto no implica ausencia de eliminaciones individuales: la métrica CER sí registra operaciones de eliminación.
- Persisten las confusiones entre caracteres visualmente similares y algunos caracteres adicionales en el reconocimiento.

**Interpretación:** es el segundo modelo con mejor precisión del conjunto. Su comportamiento es más heterogéneo y presenta menos errores de edición acumulados que ambas variantes de SVTRv2. La configuración evaluada usa tres pasos de muestreo y no ha sido ajustada con ELPD.

## 5. Comparación de patrones

| Patrón | PARSeq | SVTRv2-Tiny | SVTRv2-Small | MDiff4STR-Small |
|---|---|---|---|---|
| Confusión de caracteres similares | Presente | Presente | Presente | Presente |
| Inserción de caracteres ajenos | Frecuente | **Muy frecuente** | **Muy frecuente** | Presente |
| Repetición anómala de secuencias | No destacada | **Destacada** | **Destacada** | No destacada |
| Truncamiento grave | Puntual | Presente | Presente | **No observado en la revisión visual** |
| Salidas vacías o casi vacías | No destacado | Truncamientos observados | **Observadas** | No destacadas |
| Elementos gráficos asociados visualmente a fallos | Sí | Sí | Sí | Posible, sin patrón tan marcado |

*«No destacado» significa que no se ha identificado como patrón principal en esta revisión; no equivale a demostrar que nunca ocurra.*

## 6. Conclusiones

1. **PARSeq es la mejor referencia zero-shot** en el test auditado de ELPD, tanto en EM como en CER.
2. **SVTRv2-Tiny y Small muestran una sensibilidad notable a la longitud y a los elementos no textuales**: inserciones al principio y al final, repeticiones y, en ciertos casos, truncamientos.
3. **SVTRv2-Small mejora la precisión global de Tiny**, pero conserva las principales familias de errores.
4. **MDiff4STR-Small ofrece el segundo mejor resultado** y distribuye sus fallos entre inserciones, sustituciones y eliminaciones; no se han identificado truncamientos graves en la inspección realizada.
5. **La similitud visual entre letras y números afecta a todas las arquitecturas** y debe contemplarse en las pruebas de robustez posteriores.
6. **Quince matrículas fueron falladas por los cuatro modelos**, por lo que constituyen una muestra prioritaria para estudiar dificultades compartidas.

## 7. Limitaciones y siguientes verificaciones

- **Tamaño de muestra:** la evaluación utiliza 80 imágenes auditadas; las conclusiones no deben extrapolarse automáticamente a las 735 imágenes del dataset original ni a todas las matrículas europeas.
- **Causalidad no establecida:** la coincidencia entre escudos o distintivos y un error no prueba que sean su causa. Deben realizarse pruebas con recortes más ajustados o regiones gráficas ocultadas, preservando siempre una copia del resultado original.
- **Revisión de fallos extremos:** comprobar a resolución original los casos con repeticiones, predicciones vacías, truncamientos con confianza alta y dudas sobre la etiqueta.
- **Comparación de eficiencia:** repetir las mediciones con calentamiento, mismo batch size y temporización homogénea del pipeline completo.
- **Generalización:** contrastar los resultados con UC3M-LP, donde las etiquetas son parcialmente anonimizadas y se requieren métricas específicas no directamente comparables con EM/CER completos de ELPD.
- **Trazabilidad:** mantener las predicciones originales, las configuraciones y los hashes de checkpoints sin aplicar correcciones de formato retrospectivas a estos baselines.

---

**Conclusión final:** en la evaluación inicial sobre ELPD/Kaggle, PARSeq presenta la mayor robustez de reconocimiento completo. Los fallos más característicos de SVTRv2 son las inserciones y repeticiones asociadas visualmente a elementos adicionales de las placas, mientras que MDiff4STR-Small presenta errores de distinta naturaleza y un comportamiento global intermedio. Estos resultados proporcionan una base para el análisis comparativo, el ajuste fino y las pruebas específicas de robustez.
