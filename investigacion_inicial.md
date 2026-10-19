# Investigación inicial: OCR para matrículas españolas

> **Proyecto:** GVIS — Evaluación y optimización de arquitecturas OCR para el reconocimiento de matrículas españolas  
> **Fecha de consolidación:** 19 de octubre de 2026  

## Convenciones de evidencia y lectura

- **[P] Documentado en publicación o repositorio:** datos consignados en las fichas de revisión. Que un resultado esté publicado no implica validación experimental propia ni que haya sido reproducido aquí.
- **[Z] Auditado en ZIP en una fase anterior del proyecto:** recuentos e inspecciones consignados en el informe de datasets de Kaggle y Zenodo. En esta consolidación se han reutilizado los resultados de esas auditorías; no se ha realizado una auditoría nueva de las imágenes.
- **[V] Verificado en línea en esta consolidación:** apertura de la página pública correspondiente; verificar la página no implica haber descargado pesos, entrenado modelos ni validado licencias de imágenes de terceros.
- **[I] Interpretación, hipótesis o propuesta experimental:** razonamiento para el proyecto, aún pendiente de contrastar mediante pruebas comunes.
- **ND:** dato no disponible o no confirmado en las fuentes consultadas.
- **Referencias:** `[A1]`–`[A29]` identifican las fuentes del informe de arquitecturas; `[D1]`–`[D11]`, las del informe de datasets. La numeración interna de los informes originales se conserva dentro de cada prefijo.

## 1. Resumen del problema y objetivos

**Pregunta de investigación:** **¿Qué arquitectura OCR ofrece el mejor equilibrio entre precisión, generalización y eficiencia computacional para reconocer matrículas españolas?**

**Alcance [P]:** la entrada del sistema es una **imagen de matrícula previamente localizada y recortada** y la salida es su transcripción alfanumérica. La detección de matrículas en escenas completas no forma parte de la comparación principal. Se prioriza el formato español ordinario de siete caracteres (cuatro dígitos y tres letras), sin excluir experimentos posteriores con matrículas especiales.

**Objetivos específicos [I]:**

1. Comparar distintas familias de reconocedores (CTC, Transformer, decodificación iterativa y alternativas ligeras) bajo un protocolo homogéneo.
2. Medir **exactitud de matrícula completa**, **CER**, latencia, FPS y memoria, separando los resultados de inferencia de los costes de entrenamiento.
3. Evaluar la transferencia entre imágenes brasileñas, europeas y españolas sin mezclar identidades entre entrenamiento y prueba.
4. Estudiar la robustez ante desenfoque, iluminación, perspectiva y baja resolución, así como la mejora obtenida mediante *fine-tuning* y datos sintéticos.

**Estado de la investigación:** se han documentado **12 alternativas o estudios** y se comparan **23 configuraciones de modelos**; también se analizan **6 datasets**. No existe aún una evaluación homogénea de estos reconocedores sobre matrículas españolas completamente etiquetadas. **No se declara un ganador.**

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

### 3.1. Tabla comparativa de los seis conjuntos

> **Fuentes:** seis fichas de `Datasets.docx`. Los tamaños, formatos y condiciones de RodoSol, UFPR-ALPR, UC3M-LP y UFPR-SR-Plates proceden de sus autores **[P]**; los valores de Kaggle y Zenodo se tomaron de las auditorías de ZIP previas recogidas en dicho informe **[Z]**.

| Dataset | Procedencia y tamaño | ¿Recortadas? / texto OCR | Completitud y anonimización | Licencia / acceso | Generalización y adecuación |
|---|---|---|---|---|---|
| **D01 RodoSol-ALPR** | Brasil; 20.000 imágenes 1280×720; peajes, día/noche, lluvia; formatos tradicional y Mercosur | Recortes por cuatro esquinas; transcripción explícita en TXT | Etiqueta completa documentada; rostros difuminados; ocultación de matrícula no documentada | Académico no comercial; solicitud y acuerdo | Alto volumen, variación climática; comprobar repetición de identidades y transferencia a Europa |
| **D02 UFPR-ALPR** | Brasil; 4.500 imágenes 1920×1080; 150 vehículos, tres cámaras móviles | Recortes por anotación; matrícula y caracteres en TXT | Texto reconstruible; anonimización sistemática no documentada | Académico no comercial; solicitud y acuerdo | Útil para movimiento y cámara; separar los 30 fotogramas por vehículo |
| **D03 UC3M-LP** | España; 1.975 imágenes, 2.547 matrículas, 12.757 caracteres anotados | Recortes por polígonos/JSON; caracteres visibles | **Un dígito y una letra difuminados por matrícula**; no GT íntegro ordinario | ODbL 1.0 según repositorio | Dominio español prioritario, pero **no** válido directamente para exactitud de matrícula completa ni CER de 7 caracteres |
| **D04 ELPD (Kaggle)** | Europa; 735 imágenes en ZIP (588/74/73); resolución variable | Mayoritariamente ya recortadas; transcripción **candidata** en nombre de archivo | 0 CSV/TXT/JSON; 90 imágenes revisadas sin anonimización sistemática; GT no validado | **MIT declarada en la ficha de Kaggle**; respetar derechos sobre imágenes | 694 IDs normalizados; **15 IDs compartidos entre particiones**; limpiar antes de medir generalización europea |
| **D05 UFPR-SR-Plates** | Brasil; **100.000 imágenes**, **10.000 tracks** (5 LR + 5 HR); tradicionales/Mercosur | Imágenes de placas y esquinas; texto completo en JSON | Ground truth completo documentado; no se describe anonimización equivalente a UC3M-LP; ZIP completo no auditado | Académico no comercial; solicitud y acuerdo | Cada matrícula en una sola partición; excelente para LR/HR, sin representar 100.000 identidades independientes |
| **D06 EU License Plates Images (Zenodo)** | Europa; **53 recortes** (51 Alemania, 1 Francia, 1 Países Bajos) | Todas recortadas; nombres de archivo dan etiquetas **candidatas**, sin GT separado | 53 imágenes inspeccionadas; no ocultación sistemática observada; revisar diéresis/separadores | Licencia concreta **ND**; descarga no implica permiso general | Pequeña prueba exploratoria externa, no benchmark entre países; sin splits ni matrículas españolas documentadas |

### 3.2. Datos y problemas de anotación

**[P] RodoSol-ALPR:** los TXT proporcionan transcripción completa y esquinas para rectificar las placas. El recorte y la transcripción se pueden preparar sin utilizar un detector durante la evaluación OCR [D4].

**[P] UFPR-ALPR:** las anotaciones incluyen posición, texto e identificación de caracteres y permiten generar los recortes. Sus múltiples imágenes por vehículo requieren partición agrupada [D5].

**[P] UC3M-LP:** aunque hay anotaciones de caracteres, **el ocultamiento deliberado de una cifra y una letra por placa** impide comparar la exactitud de matrícula completa de manera convencional. Solo se plantearán métricas por posiciones visibles o protocolos específicos [D3], [D7].

**[Z] ELPD/Kaggle:** los 735 ficheros se distribuyen en `train=588`, `val=74`, `test=73`; no incluyen fichero separado de anotaciones. Se contaron 694 matrículas según identificador normalizado y 15 identificadores compartidos entre particiones; los nombres se deben revisar visualmente antes de considerarlos ground truth. La **licencia MIT** procede de la ficha mostrada en Kaggle y consignada posteriormente en el informe; esto no verifica derechos sobre cada fotografía [D8].

**[P] UFPR-SR-Plates:** las 100.000 imágenes corresponden a 10.000 secuencias, no a matrículas únicas; incluye 50.000 LR y 50.000 HR y JSON con texto completo. El repositorio establece splits 40/20/40 sin compartir matrícula [D9], [D10].

**[Z] EU Plates/Zenodo:** ZIP de versión 0.0.1 con 53 imágenes ya recortadas, sin anotaciones separadas ni splits. La inspección previa no vio anonimización sistemática, pero los nombres requieren validación (por ejemplo, letras alemanas con diéresis o separadores omitidos). La licencia específica permanece **ND** [D11].

### 3.3. Uso propuesto de los datasets **[I]**

| Objetivo | Conjunto(s) propuesto(s) | Condición antes de utilizarlos |
|---|---|---|
| Entrenamiento y prueba dentro de dominio | RodoSol-ALPR | Acceso autorizado; revisar identidad de placas/vehículos entre particiones |
| Generalización ante movimiento y cámaras | UFPR-ALPR | Mantener todas las capturas de un vehículo en una única partición |
| Robustez ante baja resolución | UFPR-SR-Plates | Mantener juntos los fotogramas del mismo track; comparar LR vs. HR |
| Transferencia a matrículas europeas | ELPD/Kaggle | Revisar texto de filenames, anonimización, calidad y 15 identidades cruzadas; rehacer splits |
| Pequeña prueba externa europea | EU Plates/Zenodo | Validar manualmente 53 etiquetas y confirmar derechos/licencia |
| Análisis del dominio español | UC3M-LP | No evaluar lectura íntegra de 7 caracteres; enmascarar posiciones ocultas |

**Brecha principal [P/I]:** los seis conjuntos no aportan todavía, en el estado actual de la revisión, un **benchmark español de lectura completa con las siete posiciones legibles y transcripciones contrastadas**. UC3M-LP es el más pertinente geográficamente, pero su anonimización impide aplicar directamente el protocolo OCR convencional.

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

### 4.3. Estado de comprobación de enlaces (19-10-2026)

**[V] Páginas públicas abiertas durante esta consolidación:** [OpenOCR](https://github.com/Topdu/OpenOCR), [documentación SVTRv2](https://github.com/Topdu/OpenOCR/blob/main/configs/rec/svtrv2/readme.md), [pesos MDiff4STR](https://huggingface.co/topdu/MDiff4STR), [PARSeq](https://github.com/baudm/parseq), [Tiny ALPR](https://github.com/tj-cvrsg/tiny_alpr), [TextSSR](https://github.com/YesianRohn/TextSSR), [OTE](https://github.com/Xu-Jianjun/OTE), [DTrOCR de terceros](https://github.com/arvindrajan92/DTrOCR), [ABINet](https://github.com/FangShancheng/ABINet), [ViTSTR](https://github.com/roatienza/deep-text-recognition-benchmark), [TRBA](https://github.com/clovaai/deep-text-recognition-benchmark), [RodoSol-ALPR](https://github.com/raysonlaroca/rodosol-alpr-dataset), [UFPR-ALPR](https://github.com/raysonlaroca/ufpr-alpr-dataset), [UC3M-LP](https://github.com/ramajoballester/UC3M-LP), [UFPR-SR-Plates](https://github.com/valfride/UFPR-SR-Plates), [EU Plates/Zenodo](https://zenodo.org/records/3967850), [repositorio REBU-Syn](https://github.com/large-ocr-model/large-ocr-model.github.io), [guía de inferencia](https://github.com/large-ocr-model/large-ocr-model.github.io/blob/main/Inference.md), [ficha de Kaggle](https://www.kaggle.com/datasets/abdelhamidzakaria/european-license-plates-dataset).

## 5. Principales dificultades, incertidumbres y cuestiones pendientes

### 5.1. Dificultades y hechos documentados

1. **Heterogeneidad de métricas [P]:** algunos trabajos reportan exactitud por palabra, otros por matrícula o por sistema completo. La propia tabla combina conjuntos chinos, texto de escenas y matrículas, por lo que no existe ranking común.
2. **Hardware y coste no comparables [P]:** las medidas se publican en GTX 1080 Ti, RTX 2080 Ti, FPGA KV260, NPU RV1106, CPU u otras GPU. A veces faltan batch, VRAM o tiempo de inferencia; no convertir esos FPS en una jerarquía universal.
3. **Reproducibilidad parcial [P]:** faltan checkpoints/implementaciones completas en MP-LPR, CSD, DTrOCR original, OTE y HMAKD; Tiny ALPR publica inferencia, pero no un fine-tuning completo de todos los módulos.
4. **Carencia de GT íntegro español [P]:** UC3M-LP enmascara una letra y un dígito por matrícula; la exactitud de cadena completa y el CER convencional no son directamente aplicables.
5. **Etiquetas europeas provisionales [Z]:** Kaggle (735) y Zenodo (53) utilizan nombres de fichero como GT candidato; falta validación manual sistemática y normalización de diacríticos, espacios y guiones.
6. **Fugas entre particiones [Z/P]:** la auditoría previa detectó 15 identificadores compartidos en Kaggle. UFPR-ALPR y UFPR-SR-Plates contienen secuencias y deben mantener agrupadas sus identidades o tracks.
7. **Licencias y privacidad [P/Z]:** RodoSol, UFPR-ALPR y UFPR-SR-Plates requieren autorización de uso académico no comercial; EU Plates/Zenodo no tiene licencia específica confirmada; MIT figura como licencia declarada del dataset de Kaggle, sin resolver derechos de terceros.

### 5.2. Hipótesis y valoraciones aún no demostradas **[I]**

- **H1:** SVTRv2-T/S puede proporcionar mejor compromiso de velocidad y exactitud que modelos iterativos mayores en matrículas cortas, pero la precisión española de cada variante es ND.
- **H2:** MDiff4STR puede ganar robustez ante caracteres ambiguos o dañados gracias al refinamiento; debe medirse cuánto cuesta cada paso adicional sobre el mismo hardware.
- **H3:** datos de diferentes países pueden favorecer la transferencia al alfabeto latino, aunque las restricciones de formato y tipografía del país pueden exigir fine-tuning especializado.
- **H4:** el entrenamiento con proporciones variables de datos reales y sintéticos (A011/A012) puede mejorar el rendimiento ante matrícula española, pero hay riesgo de producir muestras visualmente incorrectas y sesgo de formato.
- **H5:** aplicar reglas de formato español puede corregir errores `O/0`, `B/8` y similares, pero también introducir falsas correcciones; debe hacerse una ablación separada.

### 5.3. Conclusión provisional

**[P]** La revisión documenta soluciones OCR con diferentes perfiles de coste y recursos públicos; SVTRv2 y MDiff4STR son las alternativas mejor preparadas en las fichas para comenzar una evaluación reproducible. UFPR-ALPR, RodoSol y UFPR-SR-Plates documentan texto completo, mientras que UC3M-LP es crucial para estudiar el dominio español, pero anonimiza caracteres.

**[I]** La hipótesis de trabajo es que un reconocedor CTC ligero podría ofrecer un compromiso inicial competitivo y que el refinamiento iterativo podría justificar su coste en casos difíciles. **Ambas afirmaciones deben comprobarse con el mismo corpus y equipo**.

## 6. Propuesta razonada de inicio
Estado de esta sección: propuesta de trabajo [I], construida a partir de las evidencias [P] y auditorías previas [Z] recogidas en las secciones 2–5. Ningún resultado comparativo de los modelos se ha obtenido todavía en este proyecto. El objetivo inmediato no es entrenar todos los métodos, sino establecer una comparación pequeña, reproducible y ampliable.

### 6.1. ¿Con qué 2–3 arquitecturas comenzar y por qué?
Prioridad	Arquitectura y versiones	Evidencia disponible [P]	Motivo de la selección y límite de la conclusión [I]
1. Principal	SVTRv2-Tiny y Small (A09); incorporar Base como punto de escalado	5,1 / 11,3 / 19,8 M de parámetros; 201 / 189 / 143 FPS en GTX 1080 Ti, respectivamente; código, pesos y configuración de entrenamiento en OpenOCR [A19], [A20].	Referencia rápida y reproducible, basada en CTC y sin decodificación autorregresiva en inferencia. Tiny permite fijar un mínimo de coste; Small sirve para ver si más capacidad mejora el OCR. Esas velocidades no son todavía las de nuestro equipo, ni se dispone en las fichas de una exactitud española por variante.
2. Contraste arquitectónico	MDiff4STR-Small y Base (A06)	18,9 y 31,9 M de parámetros; la configuración Base-BLC comunica 97,30 % medio en seis benchmarks de texto general; código y pesos publicados [A10]–[A12].	Su decodificación iterativa permite probar si refinar caracteres ambiguos mejora el reconocimiento ante desenfoque, oclusión o baja resolución. El coste adicional debe medirse con 1, 2 y 3 pasos bajo condiciones idénticas, sin comparar directamente los 19,21 ms de su ablación con los FPS de SVTRv2 en otra GPU.
3. Referencia adicional, si el tiempo lo permite	PARSeq con un checkpoint público verificable (arquitectura utilizada en A02/A012)	Repositorio oficial con implementación y pesos [A4]; el estudio de escalado recoge un experimento con PARSeq-S y datos reales/sintéticos [A27].	Aporta una referencia Transformer con decodificación diferente de CTC y difusión. Antes de ejecutarlo hay que verificar qué variante y checkpoint reales están disponibles; no dar por hecho que reproducen exactamente el «PARSeq-S» del estudio de escalado.


**Decisión inicial [I]:** comenzar con dos familias (SVTRv2 y MDiff4STR), utilizando dos tamaños por familia cuando los checkpoints estén disponibles; añadir PARSeq como tercera arquitectura de control solo después de estabilizar el protocolo. Tiny ALPR (A010) queda para una fase posterior: las cifras de 22 FPS y 629 mW corresponden al sistema completo sobre RV1106 y su módulo LPRNet no está preparado, según la ficha, para un fine-tuning aislado inmediato [A21], [A22]. A04 (detección) y A011/A012 (generación/estudio de datos) no son reconocedores adicionales a ejecutar.

### 6.2. ¿Qué datasets utilizar inicialmente y por qué?
Etapa	Dataset	Justificación basada en las fuentes	Uso propuesto y precauciones
Principal, condicionado al acceso	RodoSol-ALPR (D01)	20.000 imágenes, transcripción completa en TXT y cuatro esquinas de matrícula [D1], [D4].	Preparar pares recorte–texto y entrenar/validar con sus particiones, previa comprobación de identidades repetidas. Su formato brasileño/Mercosur permite empezar con secuencias latinas de siete caracteres, pero no equivale a matrícula española.
Prueba externa brasileña	UFPR-ALPR (D02)	4.500 imágenes de 150 vehículos y tres cámaras, con anotaciones de caracteres [D2], [D5].	Tras entrenar con RodoSol, medir transferencia sin reajustar sobre UFPR; mantener unidos los 30 fotogramas de cada vehículo. Si en otra fase se hace fine-tuning con UFPR, reservar vehículos distintos para la prueba.
Prueba externa europea	ELPD de Kaggle (D04)	ZIP auditado: 735 imágenes; transcripción candidata en el nombre; 15 identificadores cruzan los splits publicados [Z] [D8].	Primero revisar imagen–etiqueta, normalizar separadores y eliminar o agrupar duplicados. Para transferencia pura, usar el conjunto depurado solo como prueba externa, sin entrenar ni seleccionar hiperparámetros con él; si se decide entrenar con Kaggle, rehacer las particiones por identidad. Identificar aparte el subconjunto español verificable.
Ampliación posterior de robustez	UFPR-SR-Plates (D05)	100.000 imágenes agrupadas en 10.000 tracks con 5 LR y 5 HR y texto completo en JSON; partición por matrícula [D9], [D10].	Comparar exactitud y CER en baja y alta resolución, sin tratar diez imágenes del mismo track como observaciones independientes. Incorporarlo cuando los experimentos iniciales funcionen.
Evaluación limitada o exploratoria	UC3M-LP (D03) y EU Plates/Zenodo (D06)	UC3M-LP es español, pero difumina un dígito y una letra; Zenodo contiene solo 53 recortes, con GT inferido del nombre y licencia concreta ND [D3], [D7], [D11].	En UC3M-LP evaluar solo posiciones visibles con una métrica específica; no calcular falsamente exactitud íntegra de siete caracteres. Zenodo: revisión manual de las 53 etiquetas y comprobación de derechos antes de utilizarlo como prueba externa exploratoria.


**Condición previa [P/I]:** los datasets brasileños requieren solicitud y aceptación de uso académico no comercial. Se puede probar el pipeline con una pequeña porción auditada de Kaggle.

### 6.3. ¿Qué métricas emplear para comparar los métodos?
Métricas principales [I]:
1. **Exactitud de matrícula completa (Exact Match, %):** proporción de recortes cuya transcripción normalizada coincide exactamente con la etiqueta. Es la métrica principal porque un solo carácter erróneo invalida la lectura de la matrícula. Informar siempre de N, el número de imágenes válidas evaluadas.
2. **CER (Character Error Rate):** (sustituciones + inserciones + eliminaciones) / número de caracteres del texto de referencia, acumulado sobre el conjunto. Permite diferenciar un fallo de un carácter de otro de varios. No calcularlo como CER convencional sobre las posiciones anonimizadas de UC3M-LP.
3. **Latencia y velocidad:** mediana y percentil 95 (ms/imagen) y FPS con batch = 1, indicando GPU/CPU, dispositivo, precisión numérica, versión del framework, precalentamiento y número de repeticiones; sincronizar la GPU cuando corresponda. Si se estudian lotes mayores, presentarlos por separado.
4. **Memoria y tamaño:** pico de VRAM/RAM durante inferencia, tamaño de pesos y número de parámetros. Registrar también el tiempo de fine-tuning si se entrena, pero sin confundir coste de entrenamiento con inferencia.
5. **Generalización y robustez:** reportar exactitud y CER por dataset (RodoSol → UFPR → Europa) y por condición (día/noche, lluvia, perspectiva, movimiento, resolución LR/HR), siempre que haya metadatos o una clasificación revisada. Informar de la caída respecto al dominio de entrenamiento, no solo del promedio agregado.

**Reglas comunes [I]:** fijar antes de entrenar la normalización de texto (mayúsculas, espacios, guiones y diacríticos), el alfabeto admitido y el tratamiento de casos ilegibles. Dar a cada modelo su preprocesamiento de entrada documentado, no forzar una resolución idéntica si ello perjudica artificialmente a una arquitectura. Usar el mismo hardware, los mismos recortes, las mismas particiones y un protocolo de tiempo repetible; comparar predicción OCR sin corrección de formato y, si se ensayan reglas españolas, incluirlas solo como una ablación adicional claramente etiquetada. Publicar resultados por modelo y checkpoint, no atribuir métricas generales de un artículo a todas sus variantes.


### 6.4. ¿Qué dificultades se prevén y cómo abordarlas?

| Dificultad | Evidencia o incertidumbre | Medida de mitigación propuesta [I] |
|---|---|---|
| **Falta de GT completo español** | UC3M-LP oculta deliberadamente dos caracteres [D3], [D7]. | Mantener una evaluación parcial y separada sobre UC3M-LP; buscar o construir, con permisos, una muestra española con siete caracteres legibles y etiquetas revisadas antes de concluir sobre España. |
| **Anotaciones dudosas en Europa** | Kaggle tiene etiquetas inferidas de nombres de archivo y Zenodo carece de GT formal [Z]. | Auditar imagen–texto, descartar ejemplos ambiguos, fijar normalización y conservar un registro de exclusiones/correcciones. No reportar un test como fiable antes de terminar la auditoría. |
| **Fuga entre particiones y repetición temporal** | 15 IDs cruzados en Kaggle; UFPR y UFPR-SR contienen múltiples capturas por identidad/track [Z/P]. | Separar por matrícula y, cuando sea posible, por vehículo; buscar duplicados exactos y visualmente similares; dejar el conjunto externo fuera de todo ajuste y selección del modelo. |
| **Diferencias entre formatos nacionales** | Placas brasileñas y españolas comparten caracteres latinos, pero no tipografía, distribución y patrones [D1]–[D5]. | Medir primero rendimiento *zero-shot* entre países; después evaluar el efecto de un *fine-tuning* con datos españoles íntegros solo si se consiguen. Evitar aplicar reglas brasileñas en la prueba española. |
| **Reproducibilidad y recursos** | No todos los modelos tienen pesos; el coste publicado procede de hardware dispar [A4], [A10]–[A12], [A19]–[A22]. | Verificar checkpoints e inferencia antes de seleccionar un modelo definitivamente; comenzar con variantes pequeñas, registrar versiones y usar un único equipo para medir eficiencia. Aplazar Tiny ALPR hasta aislar el módulo OCR. |
| **Coste frente a mejora real** | El refinamiento de MDiff4STR puede añadir operaciones; no hay resultados comparables propios [A10]. | Evaluar uno, dos y tres pasos sobre los mismos recortes; analizar mejora de Exact Match/CER frente a milisegundos adicionales, especialmente en imágenes difíciles. |
| **Licencia y privacidad** | Acceso no comercial en datasets UFPR/RodoSol; derechos de Zenodo sin verificar y caracteres anonimizados en UC3M-LP [D4]–[D11]. | Registrar procedencia, términos de acceso y restricciones antes de descargar/compartir datos; no redistribuir recortes, transcripciones o derivados sin permiso. |

### 6.5. ¿En qué orden realizar los primeros experimentos?

| Orden | Actividad concreta | Resultado o criterio para avanzar |
|---|---|---|
| **E1. Validar recursos** | Confirmar acceso y condiciones de RodoSol y UFPR; descargar checkpoints realmente disponibles de SVTRv2-T/S y MDiff4STR-S; ejecutar una inferencia de ejemplo con cada uno. | Inventario de checkpoints, dependencias, equipo y salida OCR. Si un modelo falla, resolverlo antes de medir. |
| **E2. Construir un corpus OCR limpio** | Extraer/rectificar recortes con las cuatro esquinas de RodoSol, unirlos a sus TXT; definir nomenclatura, normalización, filtros de ilegibilidad y particiones agrupadas. Preparar UFPR como conjunto externo separado. | Manifest reproducible con `ruta_imagen`, `texto_gt`, `dataset`, `split`, `id_grupo`, atributos y comprobaciones de duplicados. |
| **E3. Establecer una línea base sin ajuste** | Ejecutar los modelos preentrenados sin *fine-tuning* en el mismo subconjunto limpio; medir Exact Match, CER, latencia y memoria. | Primeras cifras propias, con hardware, modelo, checkpoint y N documentados; análisis de errores visibles. |
| **E4. Fine-tuning controlado** | Ajustar primero SVTRv2-T con RodoSol (mismo entrenamiento/validación para cualquier comparación posterior); repetir con SVTRv2-S y MDiff4STR-S según recursos, sin usar conjuntos externos para seleccionar hiperparámetros. | Comparativa antes/después del ajuste y curvas de validación; posibilidad de reproducir el entrenamiento. |
| **E5. Evaluación externa y por condiciones** | Aplicar los modelos fijados a UFPR-ALPR y al ELPD depurado sin nuevo entrenamiento. Analizar por cámara y calidad; evaluar UC3M-LP solo con protocolo de caracteres visibles. | Tabla cruzada por modelo/dataset y análisis de pérdida de generalización, sin afirmar exactitud española íntegra a partir de UC3M-LP. |
| **E6. Ablaciones de coste** | Comparar SVTRv2-T/S/B con el mismo test y MDiff4STR con 1/2/3 refinamientos; medir medianas/p95 de latencia y VRAM. Incorporar PARSeq como tercer control si está preparado. | Curva o tabla precisión–latencia–memoria, con condiciones equivalentes. |
| **E7. Ampliaciones justificadas** | Introducir UFPR-SR-Plates para LR/HR; estudiar datos sintéticos (TextSSR/REBU-Syn) y reglas de matrícula española solo si los resultados de E3–E6 señalan una necesidad concreta. | Experimentos aislados con grupo de control; evidencia de que cada modificación mejora realmente las métricas. |


Criterio de decisión provisional [I]. No se elegirá un «mejor modelo» por el porcentaje máximo obtenido en publicaciones distintas. Se escogerán las configuraciones no dominadas en exactitud de matrícula completa, CER, latencia y memoria, dando prioridad al comportamiento sobre datos externos no vistos.