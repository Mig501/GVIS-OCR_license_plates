# Investigación inicial: OCR para matrículas españolas

> **Proyecto:** GVIS — Evaluación y optimización de arquitecturas OCR para el reconocimiento de matrículas españolas  
> **Fecha de actualización:** 26 de octubre de 2026  

## Convenciones de evidencia y lectura

- **[P] Documentado en publicación o repositorio:** datos consignados en las fichas de revisión. Que un resultado esté publicado no implica validación experimental propia ni que haya sido reproducido aquí.
- **[Z] Auditado en ZIP en una fase anterior del proyecto:** recuentos e inspecciones consignados en el informe de datasets de Kaggle y Zenodo. En esta consolidación se han reutilizado los resultados de esas auditorías; no se ha realizado una auditoría nueva de las imágenes.
- **[V] Página pública contrastada en línea:** apertura de la página pública correspondiente; verificar la página no implica haber descargado pesos, entrenado modelos ni validado licencias de imágenes de terceros.
- **[I] Interpretación, hipótesis o propuesta experimental:** razonamiento para el proyecto, aún pendiente de contrastar mediante pruebas comunes.
- **ND:** dato no disponible o no confirmado en las fuentes consultadas.
- **Referencias:** `[A1]`–`[A29]` identifican las fuentes del informe de arquitecturas; `[D1]`–`[D14]`, las del informe de datasets. La numeración interna de los informes originales se conserva dentro de cada prefijo.

## 1. Resumen del problema y objetivos

**Pregunta de investigación:** **¿Qué arquitectura OCR ofrece el mejor equilibrio entre precisión, generalización y eficiencia computacional para reconocer matrículas españolas?**

**Alcance [P]:** la entrada del sistema es una **imagen de matrícula previamente localizada y recortada** y la salida es su transcripción alfanumérica. La detección de matrículas en escenas completas no forma parte de la comparación principal. Se prioriza el formato español ordinario de siete caracteres (cuatro dígitos y tres letras), sin excluir experimentos posteriores con matrículas especiales.

**Objetivos específicos [I]:**

1. Comparar distintas familias de reconocedores (CTC, Transformer, decodificación iterativa y alternativas ligeras) bajo un protocolo homogéneo.
2. Medir **exactitud de matrícula completa**, **CER**, latencia, FPS y memoria, separando los resultados de inferencia de los costes de entrenamiento.
3. Evaluar la transferencia entre imágenes brasileñas, europeas y españolas sin mezclar identidades entre entrenamiento y prueba.
4. Estudiar la robustez ante desenfoque, iluminación, perspectiva y condiciones ambientales adversas, y la influencia de *fine-tuning* y datos sintéticos; para LRLPR-26, respetar la selección exclusiva HR acordada.

**Estado de la investigación:** se han documentado **12 alternativas, sistemas o estudios**, y la tabla recoge **23 filas bibliográficas de modelos y variantes** (no son 23 evaluaciones experimentales); se analizan **7 datasets**. LRLPR-26 (D07) se incorpora **exclusivamente como selección de 100.000 imágenes HR de entrenamiento** (50.000 A + 50.000 B), sin incluir imágenes LR; las 50.000 HR de A ya proceden de UFPR-SR-Plates (D05). No existe aún una evaluación homogénea de estos reconocedores sobre matrículas españolas completamente etiquetadas. **No se declara un ganador.**

## 2. Arquitecturas OCR identificadas

### 2.1. Inventario y alcance de las alternativas

| Código | Arquitectura / estudio | Familia o función | Situación para este proyecto |
|---|---|---|---|
| A01 | MP-LPR | TrOCR/Transformer con cuantización mixta y FPGA | OCR de matrículas; reproducción específica difícil |
| A02 | CSD-S/B/D | ViT + decodificador por permutaciones y autodestilación | OCR; sin checkpoints propios confirmados |
| A03 | DTrOCR | Transformer exclusivamente decodificador, autorregresivo | OCR; faltan pesos originales localizados |
| A04 | FF-LPD | Detector temporal de matrículas en vídeo | **Fuera del alcance inicial:** no transcribe texto |
| A05 | Fusión OCR + VLM + reglas + LLM | Sistema híbrido y posprocesamiento por formato | Interesa el posprocesamiento; difícil comparación aislada |
| A06 | MDiff4STR-S/B | Codificador SVTRv2 + decodificación discreta por difusión | **Candidato experimental**; código y pesos publicados |
| A07 | OTE-A/OTE-N | Representación visual global + decodificación secuencial o paralela | OCR; pesos oficiales no confirmados en fichas |
| A08 | HMAKD | Destilación de múltiples profesores hacia estudiante | OCR; requiere reproducir la destilación |
| A09 | SVTRv2-T/S/B | CNN/Transformer visual + CTC | **Candidato experimental principal**; código y pesos publicados |
| A010 | Tiny ALPR | Sistema embebido; reconocimiento CNN LPRNet | Candidato condicional: aislar el reconocedor del sistema |
| A011 | TextSSR | Generación de imágenes de texto mediante difusión | **Complementario:** datos sintéticos, no OCR |
| A012 | Estudio de escalado / REBU-Syn | Experimentos con TrOCR, PARSeq y CLIP4STR; mezcla real/sintética | **Complementario:** estudio de escalado y datos; no arquitectura única |

Las alternativas A04, A011 y A012 se conservaron en la revisión, pero **no se deben clasificar como tres reconocedores OCR nuevos**. En A012 sí se comparan modelos OCR existentes y sus versiones; esas configuraciones aparecen en la tabla detallada para preservar la evidencia del documento original.

### 2.2. Tabla comparativa por modelo y versión

> **Fuente:** tabla original suministrada (`Tabla_comparacion_arquitecturas_OCR.docx`). Los resultados publicados proceden de benchmarks, tareas y dispositivos distintos; **no se deben ordenar como si fueran una única prueba comparable**. «España: ND» significa que la ficha no demuestra rendimiento en el dominio objetivo.

| Modelo | Precisión y generalización | Coste computacional | Adaptación y recursos |
| --- | --- | --- | --- |
| A01 · MP-LPR (precisión mixta) | 98,34 % top-1 en CCPD-Challenge; pruebas adversas chinas. Externa desconocida; España: ND. | 22,7 MB mixto frente a 55,5 MB FP16; pérdida 0,37 puntos; 20 img/s y 10 W en FPGA KV260. | Código/pesos MP-LPR: ND. TrOCR oficial disponible [A2]; ajuste a alfabeto español viable, reproducción de cuantización compleja. |
| A02 · CSD-S | Exactitud propia por variante: ND; evaluación general de texto en 11 benchmarks. España: ND. | 40,8 M parámetros; GFLOPs y latencia: ND. Entrenamiento con profesor, pseudoetiquetas y estudiante. | Código/pesos CSD: ND; se podría partir de PARSeq [A4], pero no es la misma implementación. |
| A02 · CSD-B | Exactitud propia por variante: ND; texto de escenas, no matrículas españolas. | 104,9 M parámetros; 40,9 GFLOPs; latencia y VRAM: ND. | Código/pesos CSD: ND. Fine-tuning viable en principio; preentrenamiento directo no confirmado. |
| A02 · CSD-D | 93,30 % de exactitud media en 11 benchmarks (RBU); sin prueba sobre matrículas españolas. | 110,2 M parámetros; 41,0 GFLOPs; latencia: ND; entrenamiento y autodestilación costosos. | Código/pesos CSD: ND; adaptación mediante PARSeq + reproducción de CSD, no ajuste directo verificado. |
| A03 · DTrOCR (inglés) | 99,6 % de exactitud por palabra en IIIT5K (ajustado con datos reales); generaliza a texto irregular; España: ND. | 128 M parámetros en configuración inglesa; GPU A100 para entrenar; velocidad de esta variante: ND. | Original sin código/pesos públicos localizados; implementación externa [A6] sin checkpoints. Ajuste OCR desde cero o parcial difícil. |
| A03 · DTrOCR (chino) | Reconocimiento de caracteres chinos; precisión específica en la ficha: ND. España: ND. | 105 M parámetros; 97,9 img/s sobre RTX 2080 Ti en benchmark chino; memoria: ND. | Código/pesos originales: ND. Se requiere adaptación del vocabulario y preentrenamiento visual. |
| A05 · OCR + VLM + reglas + LLM | UC3M-LP: 45,2 % → 88,3 % al añadir etapas; evaluación europea de 50 recortes escogidos, resultados con inconsistencias. España: ND. | CPU Core i7/16 GB: 6,8 s/imagen (UC3M-LP) y 7,5 s/recorte (EU Plates); no tiempo real. | Código global y pesos OCR/VLM: ND; Qwen2.5-0.5B disponible [A9]. Adaptación sobre todo por plantillas y reglas. |
| A06 · MDiff4STR-S | Exactitud específica Small: ND en ficha; benchmarks de texto irregular y ocluido; España: ND. | 18,9 M parámetros; latencia Small y VRAM: ND; entrenamiento con 4 × RTX 3090. | Código y pesos oficiales [A11], [A12]; fine-tuning viable y comparación de 1–3 pasos de refinamiento. |
| A06 · MDiff4STR-B (BLC) | 97,30 % media en 6 benchmarks; 88,44 % en Union14M y ≈84,3 % en OST. Sin prueba española. | 31,9 M parámetros; ablación con 3 pasos: 19,21 ms (variante/dispositivo de esa medición: ND). | Código y pesos oficiales [A11], [A12]; fine-tuning posible con recortes y transcripciones españoles. |
| A07 · OTE-A / SVTR | 96,1 % media en 6 benchmarks con Union14M-L; estudia texto sin contexto lingüístico. España: ND. | 25,2 M parámetros para configuración SVTR; latencia/FPS: ND; entrenamiento con 2 × RTX 4090. | Repo oficial anuncia publicación, sin código/pesos confirmados [A14]. Reimplementación OpenOCR [A11], checkpoint: ND. |
| A07 · OTE-N / SVTR | Exactitud propia N/SVTR: ND en ficha; decodificación paralela potencialmente menos precisa. España: ND. | Parámetros exactos de N: ND; codificador SVTR de referencia: 25,2 M. Latencia: ND. | Reimplementación OpenOCR [A11]; pesos compatibles: ND. Comparar OTE-N y OTE-A con mismos datos. |
| A08 · HMAKD (estudiante) | 96,7 % media en 6 benchmarks; caída de transferencia entre idiomas/dominios; España: ND. | Parámetros/latencia del estudiante: ND. Entrenamiento usa 4 × A100, 256 GB RAM; 12 h en ablación de ponderación. | Código/pesos HMAKD: ND. Profesores PARSeq, ABINet, ViTSTR y TRBA públicos [A4], [A16]–[A18]; destilación a implementar. |
| A09 · SVTRv2-T | Exactitud individual Tiny: ND. Estudio informa 96,57 % (6 benchmarks) y 86,14 % (Union14M) sin asignación por variante en la ficha. España: ND. | 5,1 M parámetros; 201 FPS en GTX 1080 Ti; VRAM mínima: ND. | Código y pesos oficiales OpenOCR [A11], [A20]; fine-tuning en español viable. |
| A09 · SVTRv2-S | Exactitud individual Small: ND; ver resultados generales del estudio en fila Tiny. España: ND. | 11,3 M parámetros; 189 FPS en GTX 1080 Ti; VRAM: ND. | Código y pesos oficiales OpenOCR [A11], [A20]; fine-tuning viable. |
| A09 · SVTRv2-B | Exactitud individual Base: ND; ver resultados generales del estudio en fila Tiny. España: ND. | 19,8 M parámetros; 143 FPS en GTX 1080 Ti; VRAM: ND. | Código y pesos oficiales OpenOCR [A11], [A20]; fine-tuning viable. |
| A010 · Tiny ALPR (sin cuantizar) | 98,7 % de exactitud media en CCPD (sistema completo, no OCR aislado); España: ND. | ≈1,28 M parámetros del sistema; latencia/consumo específicos sin cuantizar: ND. | Repo oficial con inferencia y modelos ONNX [A22]; fine-tuning completo no documentado. |
| A010 · Tiny ALPR (INT8, RV1106) | Exactitud independiente de INT8: ND; estudio comunica 99,23 % global en prueba real del sistema, con casos sin matrícula. | 22 FPS; 44,8 ms; 629 mW; modelos 4,22 MB en RV1106 (ARM + NPU). | Código de inferencia y ONNX [A22]; adaptar reconocedor de 66 clases y 8 caracteres; entrenamiento no reproducible directamente. |
| A012 · TrOCR-S (estudio escalado) | Exactitud propia en ficha: ND; estudio de texto general, sin prueba española. | 43,09 M parámetros; latencia inferencia: ND; GPU V100 32 GB en experimentos. | Código/pesos base TrOCR [A2]; variante exacta del estudio: ND. Fine-tuning sobre matrículas viable. |
| A012 · TrOCR-Huge (estudio escalado) | Exactitud propia en ficha: ND; no hay evaluación española. | 1.037,61 M parámetros; latencia, VRAM mínima y tamaño de pesos: ND. | Repositorio general TrOCR [A2]; pesos exactos del estudio: ND. Adaptación potencialmente costosa. |
| A012 · PARSeq-S (estudio escalado) | En experimento con datos mixtos y posterior ajuste real: 96,85 % → 97,01 % (según ficha); España: ND. | ≈22 M parámetros; mejor resultado a 32 épocas en su experimento; latencia: ND. | Implementación y pesos PARSeq [A4]; fine-tuning factible con matrículas españolas. |
| A012 · PARSeq-L (estudio escalado) | Exactitud específica: ND; texto general, no matrículas españolas. | Hasta ≈600 M parámetros para variantes grandes estudiadas; PARSeq-L converge en 5 épocas en experimento; latencia: ND. | Código de arquitectura PARSeq [A4]; pesos exactos de gran escala: ND; fine-tuning exigente. |
| A012 · CLIP4STR-L* (REBU-Syn) | 97,42 % media en 6 benchmarks de texto general; España: ND. | Coste de inferencia, parámetros y VRAM de esta versión: ND en ficha. | Repositorio de inferencia [A28]; pesos L* específicos: ND en guía revisada; adaptación requiere confirmar checkpoint. |
| A012 · CLIP4STR-B* (REBU-Syn) | Exactitud específica B*: ND en ficha; no prueba española. | Parámetros, memoria y velocidad de esta versión: ND. | Guía oficial enlaza pesos B* [A29]; inferencia disponible [A28]. Adaptación/fine-tuning: procedimiento ND. |

### 2.3. Interpretación y preselección **[I]**

- **Referencia de bajo coste:** SVTRv2-T (5,1 M parámetros, 201 FPS en GTX 1080 Ti), comparada con SVTRv2-S (11,3 M, 189 FPS) y -B (19,8 M, 143 FPS) sobre el mismo hardware citado. La exactitud individual por variante es **ND en la ficha**, aunque el estudio informa cifras agregadas.
- **Familia alternativa con refinamiento:** MDiff4STR-S (18,9 M) y -B (31,9 M). La precisión Base-BLC publicada es 97,30 % en seis benchmarks, mientras que el dato de 19,21 ms pertenece a una ablación específica, **no a una comparación directa** con SVTRv2.
- **Enfoque embebido condicional:** Tiny ALPR permite examinar eficiencia energética, pero 22 FPS/629 mW y ≈1,28 M parámetros se refieren al **sistema completo** sobre RV1106. Antes de compararlo como OCR, hay que aislar LPRNet y fijar su propio coste.
- **Recursos y factibilidad [P/V]:** el repositorio OpenOCR y la documentación de SVTRv2 están accesibles [A11], [A20]; los recursos MDiff4STR figuran en OpenOCR y Hugging Face [A11], [A12]. Esto no sustituye comprobar la compatibilidad de cada checkpoint con el alfabeto y la longitud de matrícula.

### 2.4. Reglas para interpretar los resultados publicados

1. La **exactitud por palabra de texto natural** no equivale a la exactitud de siete caracteres de matrícula.
2. El **99,23 %** del ensayo real de Tiny ALPR incluye casos sin matrícula y se refiere al sistema completo, no al OCR recortado [A21].
3. La precisión reportada para el sistema A05 sobre UC3M-LP debe tratarse con cautela: la ficha señala inconsistencias de recuento y el dataset oculta caracteres [A8], [D3].
4. Comparar **FPS medidos en GPUs diferentes** (o FPGA/NPU frente a CPU) no sirve para declarar cuál es más eficiente en el hardware del proyecto.
5. El número de parámetros, GFLOPs, tamaño de archivo y latencia son métricas distintas; donde falta información se conserva **ND**.

## 3. Datasets encontrados

### 3.1. Tabla comparativa de los siete conjuntos

> **Origen de la evidencia.** [P] procede de las fichas, publicaciones y repositorios del documento `Datasets(4).docx`; [Z] identifica auditorías previas de ZIP de Kaggle y Zenodo recogidas en ese informe, no una nueva inspección en esta consolidación. LRLPR-26 se describe mediante una **selección documental HR aún no descargada ni filtrada físicamente**. «HR» designa mayor resolución *relativa* dentro de cada track; **no significa una dimensión mínima de píxeles comprobada**.

| Código y dataset | Origen y volumen pertinente | Preparación para OCR / GT | Condiciones, splits y limitaciones | Acceso |
|---|---|---|---|---|
| **D01 · RodoSol-ALPR** | Brasil; **20.000** imágenes de peaje, día/noche, lluvia [P]. | Generar recortes desde cuatro esquinas; transcripción completa en TXT [P]. | Partición publicada por imagen; comprobar identidades repetidas. Dominio brasileño. | Académico no comercial; solicitud y acuerdo [P]. |
| **D02 · UFPR-ALPR** | Brasil; **4.500** imágenes, **150 vehículos**, tres cámaras móviles [P]. | Extraer recortes y transcripción desde TXT con caracteres [P]. | **30 fotogramas por vehículo**; separar por identidad; útil como prueba externa de movimiento/cámaras. | Académico no comercial; solicitud y acuerdo [P]. |
| **D03 · UC3M-LP** | España; **1.975 imágenes / 2.547 matrículas** [P]. | Recortes desde polígonos/JSON; solo caracteres visibles [P]. | **Un dígito y una letra difuminados**: no calcular Exact Match íntegro ni CER convencional de siete caracteres. | ODbL 1.0 declarada [P]. |
| **D04 · European License Plates (Kaggle / ELPD)** | Europa; **735 imágenes** (588/74/73) [Z]. | Mayormente recortadas; etiqueta candidata en el nombre de fichero, **no GT verificado** [Z]. | **15 IDs** aparecen en varios splits; revisar etiquetas y duplicados antes de test externo. | MIT declarada en ficha; cautela con derechos sobre imágenes [P/Z]. |
| **D05 · UFPR-SR-Plates** | Brasil; **100.000 imágenes**, 10.000 tracks (50.000 LR + 50.000 HR) [P]. | Placas extraídas, texto completo en JSON y esquinas [P]. | Partición por matrícula; su **mitad HR (50.000)** coincide con el escenario A de D07: **no es un corpus independiente**. | Académico no comercial; solicitud y acuerdo [P]. |
| **D06 · EU License Plates Images (Zenodo)** | Europa; **53 recortes** (51 Alemania, 1 Francia, 1 Países Bajos) [Z]. | Ya recortadas; nombres permiten GT provisional, sin archivo de etiquetas [Z]. | Sin splits, poca diversidad por país, normalizar diéresis y verificar transcripciones. Licencia concreta ND. | Descarga abierta; derechos de reutilización por comprobar [P/Z]. |
| **D07 · LRLPR-26 — selección HR** | Brasil; **100.000 HR** de entrenamiento seleccionadas, **20.000 tracks**, 5 HR por track (50.000 A + 50.000 B) [P/I]. | Recortes ya extraídos; texto y formato en A/B, esquinas solo A [P]. | **Excluir las 100.000 LR de entrenamiento y 15.000 LR del test oficial**. No existe test HR oficial: crear splits por track. **A solapa D05**. Etiquetado semiautomático por auditar. | Investigación académica no comercial; acuerdo y correo institucional [P]. |

### 3.2. Alcance exacto de LRLPR-26 (D07)

**Datos documentados [P].** El repositorio oficial de LRLPR-26 [D13] y el sitio de la competición [D14] indican que hay 20.000 tracks de entrenamiento: 10.000 de A (procedentes de UFPR-SR-Plates, capturas diurnas sin lluvia) y 10.000 de B (nuevas capturas con condiciones más variadas, incluida lluvia y noche). Cada track de entrenamiento tiene **5 LR y 5 HR** de la misma matrícula. El test de competición posee exclusivamente **5 LR por track**. El texto de matrícula se anotó semiautomáticamente usando recortes HR, y cabe esperar una fracción de errores pendientes de auditar.

| Componente | Tracks | Imágenes HR incluidas | Imágenes LR excluidas | Estado experimental |
|---|---:|---:|---:|---|
| Escenario A (D05/UFPR-SR-Plates) | 10.000 | **50.000** | 50.000 | HR seleccionadas; no constituyen datos nuevos frente a D05. |
| Escenario B (nuevas capturas) | 10.000 | **50.000** | 50.000 | HR seleccionadas; lluvia/noche según documentación. |
| Test oficial LRLPR-26 | 3.000 | **0** | 15.000 | Se excluye por ser exclusivamente LR. |
| **Subtotal elegido (entrenamiento original)** | **20.000** | **100.000** | **100.000** | **Seleccionadas documentalmente; requiere descarga, verificación y nuevo split HR.** |

**Decisión del proyecto [I].** Para las pruebas OCR solo se usarán las **HR de D07**. Este conjunto de 100.000 imágenes **no equivale a 100.000 matrículas diferentes** y tampoco es un conjunto ya separado en entrenamiento/validación/test HR. Tras solicitar acceso se deberá: (a) identificar los ficheros HR sin redimensionar artificialmente las LR; (b) conservar `track_id`, matrícula normalizada, escenario, número de vista y etiqueta; (c) **agrupar las cinco HR** de una matrícula en la misma partición y, si es posible, agrupar identidad de vehículo; (d) detectar coincidencias D07-A↔D05 y posibles duplicados con otros corpus; (e) auditar anotaciones; (f) crear un **test HR propio** sin fugas. Ninguna cifra experimental de la competición LR se comparará directamente con nuestros resultados HR.

**Precaución metodológica [I].** A y B fueron obtenidos con la misma cámara orientada en direcciones distintas y condiciones ambientales diferentes; una caída A→B será una señal de cambio de condiciones/escenario, **no una medida aislada y causal del efecto de la lluvia, la noche o el cambio de país**. Las vistas de cada track están correlacionadas: se deben comunicar N imágenes y N tracks, y si se calculan intervalos de confianza, agrupar el remuestreo por track.

### 3.3. Anotaciones, anonimización y reproducibilidad

- **RodoSol y UFPR-ALPR [P]:** las anotaciones originales permiten extraer recortes y su transcripción completa, pero los permisos y las particiones por identidad se revisarán antes de entrenar. En UFPR-ALPR deben permanecer agrupados los 30 fotogramas del mismo vehículo.
- **UC3M-LP [P]:** los caracteres ocultos imposibilitan validar la lectura íntegra de matrícula; utilizar exclusivamente un protocolo de posiciones visibles claramente documentado. Es el conjunto más representativo del objetivo geográfico, no el mejor benchmark para Exact Match convencional.
- **Kaggle y Zenodo [Z]:** las etiquetas procedentes de filenames son **candidatas**, no ground truth auditado. De Kaggle ya se conocen 15 IDs cruzados entre splits; Zenodo contiene solo 53 recortes y su licencia concreta está sin confirmar. Ambos requieren revisión antes de pruebas externas.
- **UFPR-SR-Plates y LRLPR-26 [P/I]:** conservaremos la distinción **D05 = dataset original LR+HR** frente a **D07 = selección HR A+B**. Si se emplean ambos en cualquier momento, D07-A/D05-HR deberán deduplicarse o excluirse de una de las ramas; nunca usar las mismas identidades para entrenar en un dataset y evaluar como «externo» en el otro. No se ha realizado aún descarga ni auditoría del ZIP de D07.

### 3.4. Uso razonado por fase [I]

| Necesidad | Dataset recomendado | Condición previa |
|---|---|---|
| **Corpus principal de recortes ya preparados (HR)** | **D07-LRLPR-26, escenarios A y B HR** | Acceso autorizado; filtrado real HR, auditoría de etiquetas, split interno por track/vehículo; evitar duplicación con D05. |
| Primer corpus de texto completo con extracción reproducible | D01 RodoSol-ALPR | Recortar y rectificar a partir de esquinas; comprobar identidades; sirve como referencia o alternativa si no se obtiene D07. |
| Evaluación externa de dispositivos/movimiento | D02 UFPR-ALPR | Reservar todos sus vehículos para prueba externa si no se usó para entrenar ni ajustar. |
| Evaluación europea independiente | D04 Kaggle ELPD | Revisar cada par imagen–etiqueta y consolidar IDs duplicados; no utilizar para seleccionar hiperparámetros si se presenta como test externo. |
| Características españolas | D03 UC3M-LP | Protocolo de posiciones visibles y resultados separados, sin Exact Match íntegro. |
| Evaluación europea exploratoria adicional | D06 Zenodo | Auditar 53 imágenes y condiciones de reutilización antes de incluirlo. |
| Experimentos de superresolución o comparación LR/HR **fuera del enfoque HR inicial** | D05 UFPR-SR-Plates | Solo en una ampliación aprobada; no duplicar D07-A y no introducir LR en el protocolo D07-HR. |

**Brecha persistente [P/I]:** ninguno de los siete conjuntos documentados constituye todavía una prueba española con **siete caracteres íntegros y anotaciones completas validadas**. D07 aumenta el volumen de entrenamiento disponible en formato latino y las condiciones de captura, pero no sustituye un test español auténtico.

## 4. Referencias científicas, repositorios y enlaces

Las referencias siguientes se han **transcrito y consolidado de los documentos originales**, manteniendo sus metadatos; no se han reconstruido autores o números de página ausentes. Se presentan DOI y enlaces públicos para facilitar su comprobación.

### 4.1. Fuentes de arquitecturas OCR (A)

- [A1] Jie Li, Dingjiang Yan, Fangzhou He, Zhicheng Dong y Mingfei Jiang. “A Mixed-Precision Transformer Accelerator With Vector Tiling Systolic Array for License Plate Recognition in Unconstrained Scenarios”. IEEE Transactions on Intelligent Transportation Systems, vol. 25, n.º 12, pp. 20280–20294, 2024. DOI: https://doi.org/10.1109/TITS.2024.3457815.
- [A2] Microsoft, “TrOCR,” UniLM, GitHub. [En línea]. Disponible: https://github.com/microsoft/unilm/tree/master/trocr. [Consultado: 14-oct-2026].
- [A3] A. Maracani, S. Ozkan, S. Cho, H. Kim, E. Noh, J. Min, C. J. Min, D. Park y M. Ozay, “Accurate Scene Text Recognition with Efficient Model Scaling and Cloze Self-Distillation”, Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14516–14526, 2025. DOI: https://doi.org/10.1109/CVPR52734.2025.01353.
- [A4] D. Bautista, “PARSeq: Scene Text Recognition with Permuted Autoregressive Sequence Models,” GitHub, 2022. [En línea]. Disponible: https://github.com/baudm/parseq. [Consultado: 14-oct-2026].
- [A5] M. Fujitake, “DTrOCR: Decoder-Only Transformer for Optical Character Recognition”, IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2024. https://ieeexplore.ieee.org/abstract/document/10483819
- [A6] A. Rajan, “A PyTorch Implementation of DTrOCR: Decoder-only Transformer for Optical Character Recognition”, GitHub, 2024. [En línea]. Disponible: https://github.com/arvindrajan92/DTrOCR. [Consultado: 14-oct-2026].
- [A7] H. Ding, J. Gao, Y. Yuan y Q. Wang, “FF-LPD: A Real-Time Frame-by-Frame License Plate Detector With Knowledge Distillation and Feature Propagation”, IEEE Transactions on Image Processing, vol. 33, pp. 3893–3906, 2024. DOI: https://doi.org/10.1109/TIP.2024.3414269.
- [A8] S. Issaoui, S. A. Alzakari, I. Saidi, R. Ejbali y A. Serir, “Hierarchical Vision–Language Fusion with Structural Constraint Reasoning for Robust Multi-Jurisdiction License Plate Recognition”, Applied Sciences, vol. 16, n.º 13, artículo 6792, 2026. DOI: https://doi.org/10.3390/app16136792.
- [A9] Qwen Team, “Qwen2.5-0.5B-Instruct,” Hugging Face, 2024. [En línea]. Disponible: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct. [Consultado: 14-oct-2026].
- [A10] Y. Du, M. Zhao, S. Fan, Z. Chen, C. Jia y Y.-G. Jiang, “MDiff4STR: Mask Diffusion Model for Scene Text Recognition”, Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, n.º 5, pp. 3705–3713, 2026. DOI: https://doi.org/10.1609/aaai.v40i5.37370.
- [A11] Topdu, “OpenOCR: An Open-Source Toolkit for General-OCR Research and Applications,” GitHub. [En línea]. Disponible: https://github.com/Topdu/OpenOCR. [Consultado: 14-oct-2026].
- [A12] Topdu, “MDiff4STR: Mask Diffusion Model for Scene Text Recognition,” Hugging Face, 2026. [En línea]. Disponible: https://huggingface.co/topdu/MDiff4STR. [Consultado: 14-oct-2026].
- [A13] J. Xu, Y. Wang, H. Xie y Y. Zhang, “OTE: Exploring Accurate Scene Text Recognition Using One Token,” en Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 28327–28336, 2024. DOI: https://doi.org/10.1109/CVPR52733.2024.02676.
- [A14] J. Xu, “OTE: Exploring Accurate Scene Text Recognition Using One Token,” GitHub, 2024. [En línea]. Disponible: https://github.com/Xu-Jianjun/OTE. [Consultado: 14-oct-2026].
- [A15] S. Chen, H. Sun y K. Xing, “Research on Heterogeneous Multi-Teacher Adaptive Distillation Methods for Scene Text Recognition”, Proceedings of the 5th International Conference on Computer, Artificial Intelligence and Control Engineering (CAICE 2026), ACM, pp. 656–663, enero de 2026. DOI: https://doi.org/10.1145/3804601.3804701
- [A16] S. Fang, “ABINet: Read Like Humans: Autonomous, Bidirectional and Iterative Language Modeling for Scene Text Recognition,” GitHub, 2021. [En línea]. Disponible: https://github.com/FangShancheng/ABINet. [Consultado: 14-oct-2026].
- [A17] R. Atienza, “ViTSTR: Vision Transformer for Fast and Efficient Scene Text Recognition,” GitHub, 2021. [En línea]. Disponible: https://github.com/roatienza/deep-text-recognition-benchmark. [Consultado: 14-oct-2026].
- [A18] CLOVA AI, “Deep Text Recognition Benchmark: What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis,” GitHub, 2019. [En línea]. Disponible: https://github.com/clovaai/deep-text-recognition-benchmark. [Consultado: 14-oct-2026].
- [A19] Y. Du, Z. Chen, H. Xie, C. Jia y Y.-G. Jiang, “SVTRv2: CTC Beats Encoder-Decoder Models in Scene Text Recognition”, Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 20147–20156, 2025. DOI: https://doi.org/10.1109/ICCV51701.2025.01874
- [A20] Topdu, “SVTRv2: Configurations, Training and Pretrained Models,” OpenOCR, GitHub. [En línea]. Disponible: https://github.com/Topdu/OpenOCR/blob/main/configs/rec/svtrv2/readme.md. [Consultado: 14-oct-2026].
- [A21] Y. Tang, S. Zhang, W. Liu, Z. Wang y J. Wang, “Ultra-Lightweight Automatic License Plate Recognition System for Microcontrollers: A Cost-Effective and Energy-Efficient Solution”, IEEE Transactions on Intelligent Transportation Systems, vol. 25, n.º 12, pp. 20419–20434, diciembre de 2024. DOI: https://doi.org/10.1109/TITS.2024.3480039
- [A22] TJ-CVRSG, “Tiny ALPR: Ultra-Lightweight Automatic License Plate Recognition System for Microcontrollers,” GitHub, 2024. [En línea]. Disponible: https://github.com/tj-cvrsg/tiny_alpr. [Consultado: 14-oct-2026].
- [A23] X. Ye, Y. Du, Y. Tao y Z. Chen, “TextSSR: Diffusion-based Data Synthesis for Scene Text Recognition”, Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 17464–17473, octubre de 2025. DOI: https://doi.org/10.1109/ICCV51701.2025.01622
- [A24] YesianRohn, “TextSSR: Diffusion-based Data Synthesis for Scene Text Recognition,” GitHub, 2025. [En línea]. Disponible: https://github.com/YesianRohn/TextSSR. [Consultado: 14-oct-2026].
- [A25] YesianRohn, “TextSSR: Pretrained Model,” ModelScope, 2025. [En línea]. Disponible: https://www.modelscope.cn/models/Yesianrohn/TextSSR. [Consultado: 14-oct-2026].
- [A26] YesianRohn, “TextSSR-F-v2.0: Synthetic Scene Text Recognition Dataset,” ModelScope, 2025. [En línea]. Disponible: https://www.modelscope.cn/datasets/Yesianrohn/TextSSR-F-v2.0. [Consultado: 14-oct-2026].
- [A27] M. Rang, Z. Bi, C. Liu, Y. Wang y K. Han, “An Empirical Study of Scaling Law for Scene Text Recognition,” en Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 15619–15629, 2024. [En línea]. Disponible: https://openaccess.thecvf.com/content/CVPR2024/html/Rang_An_Empirical_Study_of_Scaling_Law_for_Scene_Text_Recognition_CVPR_2024_paper.html
- [A28] Large OCR Model, “An Empirical Study of Scaling Law for Scene Text Recognition,” GitHub, 2024. [En línea]. Disponible: https://github.com/large-ocr-model/large-ocr-model.github.io. [Consultado: 14-oct-2026].
- [A29] Large OCR Model, “Inference Guide — CLIP4STR-B*,” GitHub. [En línea]. Disponible: https://github.com/large-ocr-model/large-ocr-model.github.io/blob/main/Inference.md. [Consultado: 14-oct-2026].

### 4.2. Fuentes de datasets (D)

- [D1] R. Laroca, E. V. Cardoso, D. R. Lucio, V. Estevam y D. Menotti, “On the Cross-dataset Generalization in License Plate Recognition,” VISAPP, pp. 166–178, 2022. DOI: https://doi.org/10.5220/0010846800003124.
- [D2] R. Laroca et al., “A Robust Real-Time Automatic License Plate Recognition Based on the YOLO Detector,” IJCNN, pp. 1–10, 2018. DOI: https://doi.org/10.1109/IJCNN.2018.8489629.
- [D3] Á. Ramajo-Ballester, J. M. Armingol Moreno y A. de la Escalera Hueso, “Dual license plate recognition and visual features encoding for vehicle identification,” Robotics and Autonomous Systems, vol. 172, art. 104608, 2024. DOI: https://doi.org/10.1016/j.robot.2023.104608.
- [D4] RodoSol-ALPR, repositorio oficial: https://github.com/raysonlaroca/rodosol-alpr-dataset
- [D5] UFPR-ALPR, repositorio oficial: https://github.com/raysonlaroca/ufpr-alpr-dataset
- [D6] UFPR-ALPR, acuerdo de licencia: https://github.com/raysonlaroca/ufpr-alpr-dataset/blob/master/license-agreement.md
- [D7] UC3M-LP, repositorio oficial y licencia ODbL 1.0: https://github.com/ramajoballester/UC3M-LP
- [D8] European License Plates Dataset, Kaggle: https://www.kaggle.com/datasets/abdelhamidzakaria/european-license-plates-dataset
- [D9] V. Nascimento, G. E. Lima, R. O. Ribeiro, W. R. Schwartz, R. Laroca y D. Menotti, “Toward Advancing License Plate Super-Resolution in Real-World Scenarios: A Dataset and Benchmark,” Journal of the Brazilian Computer Society, vol. 31, pp. 435–449, 2025. DOI: https://doi.org/10.5753/jbcs.2025.5159.
- [D10] UFPR-SR-Plates, repositorio oficial y condiciones de acceso: https://github.com/valfride/UFPR-SR-Plates
- [D11] P. Mundt, “EU License Plates Images,” Zenodo, versión 0.0.1, 30 de julio de 2020. DOI: https://doi.org/10.5281/zenodo.3967850. Disponible: https://zenodo.org/records/3967850.
- [D12] R. Laroca et al., “ICPR 2026 Competition on Low-Resolution License Plate Recognition,” *International Conference on Pattern Recognition (ICPR)*, pp. 256–275, agosto de 2026. DOI: https://doi.org/10.1007/978-3-032-31936-4_18.
- [D13] R. Laroca et al., “LRLPR-26 Dataset,” repositorio oficial, GitHub, 2026. https://github.com/raysonlaroca/lrlpr-26-dataset. Acuerdo: https://github.com/raysonlaroca/lrlpr-26-dataset/blob/main/license-agreement.pdf.
- [D14] ICPR 2026, “Competition on Low-Resolution License Plate Recognition,” página oficial y documentación de los escenarios, 2026. https://icpr26lrlpr.github.io/. [Comprobado: 26-oct-2026].

### 4.3. Estado de comprobación de los enlaces

**[V] Comprobados en la consolidación anterior (19-10-2026):** [OpenOCR](https://github.com/Topdu/OpenOCR), [SVTRv2](https://github.com/Topdu/OpenOCR/blob/main/configs/rec/svtrv2/readme.md), [MDiff4STR](https://huggingface.co/topdu/MDiff4STR), [PARSeq](https://github.com/baudm/parseq), [Tiny ALPR](https://github.com/tj-cvrsg/tiny_alpr), [TextSSR](https://github.com/YesianRohn/TextSSR), [OTE](https://github.com/Xu-Jianjun/OTE), [DTrOCR terceros](https://github.com/arvindrajan92/DTrOCR), [ABINet](https://github.com/FangShancheng/ABINet), [ViTSTR](https://github.com/roatienza/deep-text-recognition-benchmark), [TRBA](https://github.com/clovaai/deep-text-recognition-benchmark), [RodoSol-ALPR](https://github.com/raysonlaroca/rodosol-alpr-dataset), [UFPR-ALPR](https://github.com/raysonlaroca/ufpr-alpr-dataset), [UC3M-LP](https://github.com/ramajoballester/UC3M-LP), [UFPR-SR-Plates](https://github.com/valfride/UFPR-SR-Plates), [EU Plates/Zenodo](https://zenodo.org/records/3967850), [estudio de escalado](https://github.com/large-ocr-model/large-ocr-model.github.io), [guía de inferencia](https://github.com/large-ocr-model/large-ocr-model.github.io/blob/main/Inference.md) y [Kaggle ELPD](https://www.kaggle.com/datasets/abdelhamidzakaria/european-license-plates-dataset). No se han vuelto a inspeccionar todas estas páginas el 26 de octubre.

**[V] Contrastados el 26-10-2026:** [sitio ICPR 2026 LRLPR](https://icpr26lrlpr.github.io/) [D14] y [repositorio LRLPR-26](https://github.com/raysonlaroca/lrlpr-26-dataset) [D13], incluidas sus distribuciones por track, separación A/B, condición de test LR y acceso no comercial. Esto **no** significa haber descargado imágenes, ejecutado scripts ni auditado anotaciones.

## 5. Principales dificultades, incertidumbres y cuestiones pendientes

### 5.1. Dificultades apoyadas en la revisión

1. **Resultados no equiparables entre artículos [P].** Se publican accuracies de texto general, matrículas y sistemas completos, a menudo sobre GPUs, FPGA o NPU distintas. Ningún porcentaje ajeno demuestra superioridad en matrículas españolas o en nuestro equipo.
2. **Modelos con reproducibilidad desigual [P].** SVTRv2 y MDiff4STR tienen recursos públicos; MP-LPR, CSD, DTrOCR original, OTE y HMAKD presentan lagunas de checkpoints/procedimientos. Tiny ALPR exige aislar OCR del detector y verificar su adaptación.
3. **Falta de ground truth español íntegro [P].** UC3M-LP difumina un dígito y una letra; no sirve directamente como test de lectura íntegra de siete caracteres.
4. **Etiquetado europeo provisional y condiciones de uso [Z/P].** En Kaggle se derivan transcripciones del nombre y aparecen 15 IDs cruzados entre particiones; en Zenodo los 53 nombres tampoco son GT revisado y los derechos específicos permanecen ND.
5. **Riesgo de fuga por secuencia [P].** UFPR-ALPR tiene imágenes repetidas por vehículo; D05 y D07 tienen varias vistas por track. Los splits deben hacerse por track y, cuando se pueda, por matrícula/vehículo, no por imagen.
6. **Solapamiento entre datasets [P].** El escenario A de D07 reutiliza las 10.000 secuencias de UFPR-SR-Plates, incluidas **50.000 HR**. D05 y D07 no son dos dominios independientes para esas muestras; habrá que cruzar identificadores/archivos antes de construir las particiones.
7. **Ausencia de test oficial HR en LRLPR-26 [P].** El test publicado contiene 15.000 LR, **excluidas** por decisión del proyecto. Las 100.000 HR elegidas provienen del entrenamiento original, así que su evaluación exige un **test HR propio**, con separación correcta y criterio de no contaminación.
8. **Calidad de anotación D07 por auditar [P].** El etiquetado de matrículas se generó semiautomáticamente usando HR. Esto da etiquetas documentadas, **no garantía de que cada transcripción sea correcta**; comprobar muestras estratificadas de A/B y preservar un log de correcciones/exclusiones.
9. **Confusión HR con píxeles fijos [P/I].** En LRLPR-26 HR significa la vista relativamente más nítida/cercana del track, no un umbral único de anchura o altura. No se deben inventar resoluciones mínimas ni etiquetar como HR imágenes ampliadas digitalmente desde LR.
10. **Acceso, licencias y privacidad [P].** D01, D02, D05 y D07 requieren condiciones académicas no comerciales; no compartir imágenes ni derivados fuera del acuerdo. El uso abierto de Kaggle no aclara por sí solo todos los derechos de terceros.

### 5.2. Hipótesis y preguntas a contrastar **[I]**

- **H1 (coste):** SVTRv2-Tiny/Small podría ofrecer buena relación Exact Match–latencia frente a arquitecturas con refinamiento, pero faltan cifras comparables propias.
- **H2 (refinamiento):** MDiff4STR podría corregir caracteres ambiguos de recortes HR con desenfoque/perspectiva; la ganancia tal vez no compense 2–3 pasos extra en matrículas cortas.
- **H3 (escenario):** un modelo ajustado con HR del escenario A puede perder exactitud sobre HR del escenario B por el cambio conjunto de cámara/orientación y condiciones ambientales; es una hipótesis sobre **A↔B**, no sobre países distintos.
- **H4 (transferencia):** entrenar con matrículas brasileñas puede transferir al alfabeto latino europeo, pero las diferencias de estilo y secuencia exigen medir pruebas externas y adquirir GT íntegro español.
- **H5 (más datos vs. limpieza):** usar las 50.000 HR nuevas de B podría aportar más diversidad, siempre que la revisión de etiquetas y separación por identidad compense posibles errores de pseudoetiquetado.
- **H6 (datos sintéticos y reglas):** TextSSR/REBU-Syn y las restricciones del formato español podrían mejorar algunos errores, pero también crear etiquetas falsas o sobrecorrecciones; se estudiarán solo como ablaciones posteriores.

### 5.3. Estado crítico de la evidencia

**[P]** La documentación identifica candidatos reproducibles (SVTRv2, MDiff4STR), conjuntos con texto íntegro (RodoSol, UFPR-ALPR y D07) y un conjunto español valioso pero parcialmente anonimizado (UC3M-LP). **[V]** El material público de LRLPR-26 describe exactamente 100.000 HR de entrenamiento seleccionables en A+B y ninguna HR en su test oficial. **[I]** La prioridad operativa es lograr un corpus HR limpio, con partición fiable, antes de entrenar o comparar modelos. **No hay resultados experimentales propios ni evidencia suficiente para seleccionar un ganador.**

## 6. Propuesta razonada de inicio y plan de trabajo

> **Estado [I].** Esta sección contiene decisiones y procedimientos **propuestos**, no ensayos realizados. Su base son las fichas de arquitectura [P], las auditorías anteriores de dos ZIP [Z] y la documentación oficial actualizada de D07 [V]. La selección de LRLPR-26 está limitada **estrictamente a las 100.000 imágenes HR de entrenamiento de A+B**, y no se ha filtrado físicamente aún. La prueba HR que se plantea será **propia** y nunca se presentará como resultado del benchmark oficial LR.

### 6.1. ¿Con qué 2–3 arquitecturas comenzar y por qué?

| Prioridad | Arquitectura y versiones | Evidencia bibliográfica [P] | Razón y límite de la propuesta [I] |
|---|---|---|---|
| **1. Principal** | **SVTRv2-Tiny y Small** (A09); Base en ablación. | 5,1 / 11,3 / 19,8 M parámetros y 201 / 189 / 143 FPS publicados en GTX 1080 Ti; código/pesos en OpenOCR [A19], [A20]. | Empezar por un OCR CTC rápido y reproducible. Comparar Tiny frente a Small para ver si la capacidad adicional mejora HR-A/HR-B. Los FPS publicados **no son** medidas de nuestro equipo ni su precisión sobre matrículas españolas. |
| **2. Contraste arquitectónico** | **MDiff4STR-Small**, después Base (A06). | 18,9 / 31,9 M parámetros; pesos e implementación [A10]–[A12]; Base-BLC 97,30 % medio en seis benchmarks de texto general. | Comprobar si refinar predicciones (1/2/3 pasos) reduce errores en HR con degradaciones reales a un coste aceptable. No comparar directamente cifras de artículos tomadas sobre hardware distinto. |
| **3. Control opcional** | **PARSeq** con checkpoint verificable [A4]. | Repositorio oficial y checkpoints; familia Transformer con decodificador distinta de CTC y difusión, utilizada en el estudio [A27]. | Aportar una tercera referencia solo si la instalación e inferencia son reproducibles; no afirmar que el checkpoint disponible reproduce exactamente PARSeq-S de A012. |

**No prioritarias por ahora [I]:** Tiny ALPR requeriría separar LPRNet de su pipeline completo; FF-LPD solo detecta; TextSSR genera datos; A012 es un estudio de modelos y mezclas de datos, **no** un OCR independiente. Se conservan en la revisión por su posible uso posterior.

### 6.2. ¿Qué dataset o datasets utilizar inicialmente y por qué?

| Papel propuesto | Dataset / subconjunto | Justificación | Uso y precaución [I] |
|---|---|---|---|
| **Principal HR, condicionado a permiso** | **D07 · LRLPR-26, solo 100.000 HR A+B** | Recortes existentes con texto de matrícula; A: 50.000 HR diurnas controladas; B: 50.000 HR con condiciones más variadas [D12]–[D14]. | **Crear split propio por track**, auditar GT y no utilizar 100.000 LR de train ni 15.000 LR de test. Comparar resultados globales y por escenario; no entrenar en D05 y presentar D07-A como test externo. |
| **Referencia/fallback de entrenamiento OCR completo** | **D01 · RodoSol-ALPR** | 20.000 imágenes con texto completo y cuatro esquinas [D1], [D4]. | Extraer/rectificar recortes de forma reproducible; reservar identidades para prueba. Si D07 tarda en estar disponible, empezar el pipeline y baseline con RodoSol (o un subconjunto permitido). |
| **Test externo brasileño** | **D02 · UFPR-ALPR** | 4.500 imágenes de 150 vehículos y tres cámaras, con anotaciones [D2], [D5]. | Mantener las capturas de cada vehículo juntas. Solo es externo si ese vehículo/placa no se ha usado en entrenamiento, validación o selección de hiperparámetros; auditar cruces si se mezclan fuentes. |
| **Test europeo sin ajuste** | **D04 · ELPD/Kaggle** | 735 imágenes europeas, varias ya recortadas, con etiquetas candidatas [Z]. | Revisar los 735 pares imagen–texto, las 15 coincidencias de IDs entre splits y duplicados; mantener el conjunto depurado fuera del ajuste de modelos para llamarlo evaluación externa. |
| **Análisis del dominio español** | **D03 · UC3M-LP** | 2.547 matrículas españolas, con caracteres parcial y deliberadamente difuminados [D3], [D7]. | Protocolo restringido a posiciones visibles; no comunicar Exact Match íntegro ni CER convencional de 7 caracteres. |
| **Complemento europeo pequeño** | **D06 · Zenodo** | 53 recortes europeos [Z]. | Auditar totalmente las etiquetas y confirmar licencia antes de inferencias/publicación. |
| **Relacionado, NO conjunto independiente del HR-A** | **D05 · UFPR-SR-Plates** | Escenario A de D07 procede de él [D9], [D10], [D13]. | Conservar trazabilidad de sus 50.000 HR compartidas; no duplicarlas. Un posible estudio LR/HR con D05 sería un **experimento futuro distinto** del protocolo actual D07-HR. |

**Elección operativa [I]:** priorizar el dataset D07-HR en cuanto el acceso esté aprobado, porque ya ofrece recortes y variación ambiental. No supeditar toda la reproducibilidad al permiso: preparar el código de anotaciones, la infraestructura de evaluación y un conjunto piloto con RodoSol/Kaggle revisado según derechos, sin contaminar la prueba europea. Crear dos vistas de la evaluación: una **interna HR** por track para medir A/B y una **externa** por país/fuente (UFPR, Europa), usando modelos congelados.

### 6.3. ¿Qué métricas se emplearán?

| Dimensión | Métrica y definición [I] | Protocolo |
|---|---|---|
| **Principal OCR** | **Exact Match / exactitud completa**: proporción de imágenes donde la cadena predicha normalizada coincide exactamente con GT. | Por recorte HR, conjunto, escenario A/B y modelo. Informar de número de imágenes **y tracks**. |
| **Error por carácter** | **CER**: suma de distancias de edición (sustituciones + inserciones + eliminaciones) / suma de longitudes GT. | No mezclar CER íntegro con métricas parciales de UC3M-LP. Documentar las reglas de normalización. |
| **Eficiencia** | **Latencia mediana y p95** (ms/imagen), **FPS**, memoria pico de RAM/VRAM, parámetros y tamaño del checkpoint. | Mismo hardware, batch=1, precalentamiento, sincronización GPU y repeticiones; indicar precisión FP32/FP16 y coste de preprocesamiento. No reutilizar FPS publicados como resultado propio. |
| **Robustez de escenario** | Exact Match y CER estratificados por **A/B** y por condiciones auditadas (lluvia, noche, desenfoque, etc.). | Las condiciones individuales se compararán únicamente si están anotadas o se revisan manualmente. No atribuir diferencias A/B a una única causa. |
| **Generalización** | Exact Match/CER en **test HR interno** y en **UFPR, Kaggle y UC3M-LP con protocolo adecuado**; caída relativa/absoluta. | Congelar modelos antes del test externo; declarar diferencia de procedencia, fuentes potencialmente repetidas y tamaños. |
| **Análisis por secuencia** | **Exact Match por track** (opcional): acierto tras combinar cinco predicciones HR mediante método predefinido. | Reportar aparte del OCR de una sola imagen; incluir coste de cinco inferencias y latencia por track. No equivale a la competición oficial LR. |

**Reglas de evaluación [I]:** definir antes del entrenamiento alfabeto, uso de mayúsculas, eliminación de guiones/espacios/diacríticos, normalización de caracteres españoles y tratamiento de etiquetas dudosas; usar **el mismo contenido visual** para todos los modelos, permitiendo el preprocesamiento/resolución de entrada exigido por cada arquitectura. Guardar metadatos, semillas, checkpoint y versión del entorno. Para incertidumbre estadística, preferir análisis y bootstrap agrupados por *track* frente a interpretar 100.000 vistas como independientes. No optimizar sobre test externo.

### 6.4. ¿Qué dificultades se prevén y cómo abordarlas?

| Dificultad | Evidencia o incertidumbre | Medida de mitigación propuesta [I] |
|---|---|---|
| **No existe test oficial HR en LRLPR-26** | Las 100.000 HR proceden solo del entrenamiento original; test oficial contiene exclusivamente LR [D13], [D14]. | Crear train/val/test **propios** en HR agrupando 5 vistas de cada track e identidad cuando conste. Registrar protocolo y no comparar sus cifras con la tabla de competición LR. |
| **Solapamiento D07-A y D05** | 10.000 tracks de D07-A ya estaban en UFPR-SR-Plates: **50.000 HR** compartidas [D9], [D10], [D13]. | Establecer IDs canónicos y hashes; desduplicar y no usar D05-HR como test independiente de un modelo entrenado en D07-A. |
| **GT de D07 potencialmente imperfecto** | Etiquetado semiautomático con modelo aplicado a HR [D14]. | Auditar muestra aleatoria estratificada por A/B y casos difíciles; corregir/excluir con historial de decisiones; verificar una submuestra del futuro test HR de forma especialmente estricta. |
| **Falta de GT español completo** | UC3M-LP difumina un dígito y una letra [D3], [D7]. | Evaluación separada por caracteres visibles; buscar/construir, con autorización, un pequeño test español íntegro y manualmente revisado antes de concluir sobre el objetivo geográfico. |
| **Anotaciones dudosas en Europa** | Kaggle usa filenames y tiene 15 IDs repetidos; Zenodo carece de etiquetas formales [Z]. | Auditar imagen–texto, normalización, ilegibilidad y duplicados; test europeo congelado, con log de exclusiones. |
| **Fuga por múltiples capturas** | Cinco HR por track de D07 y 30 imágenes por vehículo en UFPR [P]. | Partición por track, matrícula y vehículo (según metadatos); comprobar similitud visual y no seleccionar modelos con el test. Informar resultados por imagen y track. |
| **Cambio de dominio A/B y Brasil/España** | A y B difieren en condiciones de captura y orientación; España difiere en tipografía y estructura de matrícula [D13]. | Evaluar A/B por separado y transferir a UFPR/Europa sin ajuste. No atribuir causalidad ni afirmar exactitud española íntegra desde UC3M-LP. |
| **Recursos, reproducibilidad y coste** | Modelos y benchmarks usan hardware heterogéneo; algunos checkpoints son ND [A4], [A10]–[A12], [A19]–[A22]. | Validar inferencia primero; empezar con Tiny/Small; mismo hardware, semillas, dependencias y protocolo batch=1; diferir modelos difíciles y medir la mejora real por ms/MB adicional. |
| **Licencias y privacidad** | RodoSol/UFPR/D07 requieren autorización no comercial; derechos Zenodo no confirmados. | Registrar acuerdos, accesos y origen; no redistribuir imagen/recorte/checkpoint derivado si los términos no lo permiten. |

### 6.5. ¿En qué orden realizar los primeros experimentos?

| Orden | Actividad concreta | Resultado verificable / criterio para avanzar |
|---|---|---|
| **E1 · Acceso y entorno** | Solicitar **D07 LRLPR-26** (acuerdo institucional) y D01/D02 si procede. Confirmar hardware, versiones y licencias. Ejecutar una inferencia de SVTRv2-T, SVTRv2-S y MDiff4STR-S con checkpoints públicos. | Inventario `modelos_checkpoints.csv`, entorno reproducible, OCR de ejemplo para los modelos realmente ejecutables. **Si D07 no está disponible, usar D01 como primer corpus con permiso.** |
| **E2 · Extraer exclusivamente HR y auditar** | Del entrenamiento D07 seleccionar **5 HR × 20.000 tracks**; excluir LR de A/B y todo el test LR oficial. Generar manifiesto con IDs, escenario, vista, GT, formato, ruta, calidad y procedencia. Muestrear y revisar GT. | Recuento esperado **50.000 HR A + 50.000 HR B** antes de exclusiones documentadas; log de auditoría, integridad de archivos y registro de permisos. |
| **E3 · Split HR sin fugas** | Construir train/val/test propios sobre **tracks/identidades**, estratificados si es posible por escenario y formato. Como reparto de partida **propuesto, no oficial**, estudiar 70/15/15 % por track (aprox. 14.000/3.000/3.000 tracks, 70.000/15.000/15.000 HR antes de exclusiones), reajustándolo si fuera necesario para agrupar identidades y evitar fugas. Identificar coincidencias **D07-A↔D05-HR**; revisar duplicados internos y cruces con datasets externos. Fijar normalización. | `manifest_hr.csv` y `splits_hr.csv` con hash/semilla/reglas; cero tracks compartidos; N de imágenes y tracks por A/B/split; reservar test sin tocar. |
| **E4 · Línea base zero-shot** | Evaluar SVTRv2-T/S y MDiff4STR-S **sin fine-tuning** en la **validación HR interna**, separando A/B; medir Exact Match, CER, mediana/p95, FPS y VRAM. **No abrir aún el test HR reservado.** Registrar fallos y errores OCR. | Tabla propia `baseline_hr.csv` con hardware/versión/modelo/N tracks y N imágenes. **El test sellado no se usa para elegir hiperparámetros.** |
| **E5 · Fine-tuning controlado** | Entrenar primero SVTRv2-T con HR de **train** (A+B) y validar únicamente en **val**; repetir con Small y MDiff4STR-S según coste. Comparar con ajustes usando solo A o A+B, sin mezclar identidades. | Checkpoints, curvas de entrenamiento, criterio de selección predefinido y comparación antes/después. Controlar A→B sin declarar independencia causal. |
| **E6 · Test interno y generalización externa** | Congelar modelos y evaluar **una sola vez** sobre **test HR** por A/B. Después probar UFPR-ALPR y ELPD depurado (y UC3M-LP solo posiciones visibles); revisar duplicados interdataset. | Matriz modelo × dataset/escenario; Exact Match, CER, intervalos por track, diferencias de generalización y análisis de errores. No confundir test HR con ranking oficial LR. |
| **E7 · Ablaciones y siguientes decisiones** | Medir SVTRv2-T/S/B, MDiff4STR con 1/2/3 pasos; introducir PARSeq solo si listo. Explorar votación de las **cinco vistas HR** como experimento independiente; añadir síntesis/reglas españolas solo si hay justificación. | Curva precisión–latencia–memoria, coste por track, tabla de ablaciones con grupo de control y conclusión fundamentada sobre qué arquitectura merece seguir optimizando. |

**Criterio de decisión [I].** No se elegirá el «mejor OCR» a partir de porcentajes de diferentes artículos ni de las 100.000 imágenes vistas como independientes. Se buscarán alternativas no dominadas en **Exact Match, CER, latencia y memoria** para el **mismo test HR propio**, valorando la transferencia externa y dejando explícito que **la superioridad sobre matrículas españolas permanece pendiente de un GT español íntegro**.
