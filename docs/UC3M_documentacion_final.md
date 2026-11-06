# UC3M-LP — Documentación final de preparación para OCR

**Proyecto:** Prácticas GVIS — Evaluación y optimización de arquitecturas OCR para el reconocimiento de matrículas.  
**Dataset:** UC3M-LP (Universidad Carlos III de Madrid).  
**Fecha de cierre de esta preparación:** 6 de noviembre de 2026.  
**Estado:** **PREPARACIÓN CERRADA. APTO PARA EXPERIMENTOS OCR CON EVALUACIÓN PARCIAL**. Integridad automatizada comprobada en todas las imágenes y revisión visual muestral completada.  
**Uso previsto:** evaluación de modelos preentrenados (*zero-shot*) y análisis de robustez sobre matrículas españolas recortadas; no constituye un test de transcripción íntegra.

## 1. Resumen ejecutivo

Se descargó y validó UC3M-LP, un conjunto de fotografías de vehículos con anotaciones de matrículas y caracteres parcialmente anonimizados. Se partió de **1.975 fotografías JPG y 1.975 anotaciones JSON**, repartidas en `train` y `test`, y se obtuvieron **2.547 recortes de matrícula** (*License Plate Only*, LPO), conservando el reparto original:

| Partición | Fotografías JPG + JSON | Recortes LPO PNG |
|---|---:|---:|
| `train` | 1.580 pares | **2.056** |
| `test` | 395 pares | **491** |
| **Total** | **1.975 pares** | **2.547** |

La generación de recortes terminó con **0 incidencias**. La auditoría posterior verificó **los 2.547 PNG mediante decodificación completa**, contrastó cada archivo con el manifiesto y registró **0 incidencias**. Se inspeccionaron visualmente **60 recortes**: 20 de entrenamiento, 20 de test y 20 casos difíciles. No se observaron errores evidentes de localización ni caracteres cortados en estas muestras.

**Limitación esencial:** las **2.547 etiquetas son parciales** (`partial_masked`) por contener posiciones anonimizadas indicadas con `*`. Por tanto, **no deben utilizarse directamente para calcular Exact Match (EM) o Character Error Rate (CER) completos**. Es necesario un protocolo específico que conserve las posiciones conocidas y controle los problemas de alineación.

## 2. Procedencia y alcance

| Campo | Información |
|---|---|
| Fuente | [UC3M-LP, Zenodo](https://zenodo.org/records/17152029) |
| Archivo original | `UC3M-LP.zip` |
| Descarga de referencia | `https://zenodo.org/records/17152029/files/UC3M-LP.zip?download=1` |
| Tamaño del ZIP registrado | **4.547.680.070 bytes** (aprox. 4,24 GiB) |
| MD5 registrado del ZIP | `496e3b5e66b78d231f71d85c0e5e11a3` |
| Contenido contabilizado | 1.975 JPG, 1.975 JSON y 2 TXT (`train.txt`, `test.txt`); **3.952 archivos** |
| Naturaleza de las imágenes | Fotografías completas de vehículos, en ocasiones con más de una matrícula anotada |
| Objetivo de esta preparación | Obtener imágenes LPO trazables para inferencia OCR sin modificar las originales |
| Licencia / condiciones de reutilización | **Pendiente de verificar expresamente en la ficha y archivos de la fuente** antes de redistribuir imágenes o autorizar usos adicionales |

La **partición `train`/`test` es la proporcionada por el dataset de origen**. No se ha construido una nueva separación por matrícula, vehículo o identidad. Tampoco se ha demostrado ausencia de vehículos repetidos entre particiones, por lo que debe verificarse cualquier posible fuga si se plantea entrenamiento o ajuste con `train`.

## 3. Formato de las anotaciones originales

Cada JSON incluye, entre otros campos:

- `imagePath`: nombre de imagen declarado en el JSON (véase la incidencia de correspondencia de la sección 5.1).
- `imageWidth`, `imageHeight`: dimensiones de la fotografía.
- `lps`: lista de matrículas presentes en la fotografía.
- Para cada matrícula: `lp_id`, `poly_coord` y `characters`.
- `poly_coord`: contorno de la matrícula, con **4, 5 o 6 vértices** observados.
- `characters`: caracteres conocidos, identificados mediante `char_id`, y sus cajas `bbox_coord`.

El valor `lp_id` incorpora un prefijo de anotación, por ejemplo `AD-2*28*GG` o `AN-074*C*V`. La parte anterior al primer guion **no se incluye** en el texto objetivo del OCR. Los asteriscos representan posiciones de matrícula anonimizadas; **no son caracteres para que el modelo intente predecir como ground truth**.

Ejemplo de normalización de `AD-2*28*GG`:

| Campo del manifiesto | Valor | Interpretación |
|---|---|---|
| `text_original` | `AD-2*28*GG` | Identificador de la anotación fuente |
| `text_normalized` | `2*28*GG` | Secuencia con posiciones anonimizadas |
| `text_visible` | `228GG` | Solo los caracteres conocidos, **sin información posicional suficiente por sí sola** |
| `known_mask` | `1011011` | `1` = posición conocida; `0` = posición anonimizada |
| `label_status` | `partial_masked` | No existe transcripción íntegra fiable |

**Regla:** preservar la secuencia con `*` y `known_mask`; no rellenar huecos, no inferir caracteres ocultos y no eliminar los asteriscos antes de una comparación posicional.

## 4. Incidencias detectadas y decisiones adoptadas

### 4.1. Campo `imagePath` inconsistente

La auditoría de correspondencia sobre las **1.975 anotaciones JSON** encontró:

- **1.975/1.975** valores de `imagePath` distintos del nombre base del JSON.
- **671** referencias de `imagePath` sin imagen correspondiente.
- **1.975/1.975** fotografías JPG homónimas del JSON disponibles en su misma partición.
- Coincidencia de dimensiones declaradas en el JSON con los JPG homónimos en los **1.975 pares**.

Las revisiones visuales de muestras de `train` y `test` confirmaron que los polígonos correspondían a la fotografía con **el mismo nombre base que el JSON**, no a la ruta indicada por `imagePath`.

**Decisión implementada:** asociar siempre `UC3M-LP/<split>/<id>.json` con `UC3M-LP/<split>/<id>.jpg`. **No utilizar `imagePath`** para generar los recortes. Esta conclusión combina validación sistemática de existencia/dimensiones y comprobación visual muestral, no una inspección humana exhaustiva de todas las parejas.

### 4.2. Polígonos con 5 y 6 vértices

La primera validación geométrica asumía incorrectamente que todas las matrículas tendrían cuatro vértices y señaló **142 casos**. Un diagnóstico posterior confirmó que eran contornos con un número de puntos distinto, no necesariamente incorrectos:

| Número de vértices | Matrículas |
|---|---:|
| 4 | 2.405 |
| 5 | 105 |
| 6 | 37 |
| **Total** | **2.547** |

Los **142 contornos de 5/6 vértices** se conservaron. No se identificaron puntos fuera de los límites de la imagen en ese análisis; algunos contornos tenían vértices repetidos, circunstancia que tampoco impedía construir una caja envolvente válida. La revisión visual de los casos muestreados mostró anotaciones útiles de matrícula.

**Decisión implementada:** calcular el rectángulo envolvente usando **todos los vértices** (`min(x)`, `min(y)`, `max(x)`, `max(y)`), sin forzar un polígono cuadrilateral ni descartar los 142 ejemplos.

## 5. Preparación LPO definitiva

Script principal: `scripts/uc3m/prepare_uc3m_lpo.py`.

Proceso aplicado:

1. Lectura directa del ZIP original, sin descomprimir todas las fotografías a disco.
2. Asociación de cada JSON con el JPG homónimo, dentro de `train` o `test`.
3. Validación de que las dimensiones de la imagen coincidan con el JSON.
4. Para cada matrícula, cálculo de la caja envolvente de **todos** los vértices.
5. Aplicación de un margen del **4 % del ancho y 4 % del alto del rectángulo** por cada lado, limitado a los bordes de la fotografía.
6. Recorte sobre los píxeles originales y guardado en **PNG**, evitando una nueva compresión con pérdida.
7. Escritura de un registro por matrícula en `manifest.csv`, con rutas de origen, transcripción parcial, máscara, geometría y partición.

**No se aplicaron:** corrección de perspectiva, escalado, filtros de iluminación, enfoque, reconstrucción de caracteres ni edición de las zonas anonimizadas. El recorte por caja envolvente puede incluir fondo del vehículo cuando la placa está inclinada; es una decisión intencional de esta versión de referencia.

**Resultado registrado:** 1.975 JSON inspeccionados, 1.975 fotografías procesadas, **2.547 PNG generados**, `train=2.056`, `test=491`, **0 incidencias** y **0 recortes previamente existentes reutilizados** durante esa ejecución. Ni los JPG/JSON del ZIP ni el archivo ZIP se modificaron.

## 6. Auditoría final e inspección visual

Script: `scripts/uc3m/audit_uc3m_lpo_final.py`.

### 6.1. Auditoría automática exhaustiva de archivos

| Control | Resultado |
|---|---:|
| Filas en `manifest.csv` | **2.547** |
| PNG abiertos y decodificados completamente | **2.547/2.547** |
| Recortes de entrenamiento | 2.056 |
| Recortes de test | 491 |
| Polígonos de 4 / 5 / 6 vértices | 2.405 / 105 / 37 |
| Etiquetas `partial_masked` | **2.547** |
| Incidencias registradas | **0** |
| Ancho mínimo / mediana / máximo | 68 / 585 / 3.715 px |
| Alto mínimo / mediana / máximo | 20 / 180 / 2.792 px |

La comprobación incluyó existencia de rutas, apertura y decodificación completa, formato PNG, dimensiones reales frente al manifiesto, unicidad de `image_path`, coherencia de particiones y nombres, máscaras de posiciones, contadores y método de recorte declarado.

### 6.2. Revisión visual muestral

Se generaron y revisaron **60 muestras**, distribuidas en:

| Hoja de contacto | Muestras | Criterio |
|---|---:|---|
| `uc3m_lpo_train_audit.jpg` | 20 | Muestra de `train` estratificada por número de vértices |
| `uc3m_lpo_test_audit.jpg` | 20 | Muestra de `test` estratificada por número de vértices |
| `uc3m_lpo_edge_cases.jpg` | 20 | Ancho/alto reducidos y proporciones extremas |

Se observaron matrículas localizadas correctamente, con caracteres visibles y bloques de anonimización conservados. **No se encontraron recortes claramente incorrectos en las muestras revisadas**. Se observaron casos con resolución muy reducida, inclinación o perspectiva pronunciadas, fondo alrededor de la placa y bajo contraste/iluminación difícil; se conservaron para estudiar robustez.

**Alcance del cierre:** la auditoría de integridad es exhaustiva sobre los **2.547 PNG**; la revisión visual corresponde a **60 entradas seleccionadas**, por lo que no certifica la calidad visual de absolutamente todos los recortes ni la exactitud independiente de todos los caracteres visibles.

## 7. Utilización para baselines y evaluación

### 7.1. Dataset que deben consumir los modelos

- **Entrada OCR:** `image_path` resuelto respecto a `/home/miguel.sanchez/datasets/ocr_matriculas/processed/UC3M-LP_LPO/`.
- **Evaluación externa UC3M-LP:** seleccionar `scenario == "test"` (**491 imágenes**).
- **Entrenamiento:** `scenario == "train"` contiene **2.056 recortes**, pero las transcripciones son parciales; **no tratar estos ejemplos como etiquetas completas para entrenamiento supervisado convencional**.
- Mantener trazabilidad de `source_path`, `annotation_path`, `view_id` y `known_mask` en los resultados de inferencia.

### 7.2. Métricas: qué es válido y qué queda pendiente

**No es correcto** comparar la predicción completa contra `text_visible` (se han eliminado los huecos), ni reemplazar `*` por caracteres supuestos, ni calcular CER/EM completos como si el texto de referencia fuera íntegro.

Propuesta para el protocolo de evaluación, **todavía pendiente de implementar y validar**:

1. **EM parcial estricto:** tras normalizar la predicción y comprobar la longitud esperada para cada muestra, marcar acierto solo cuando **todas las posiciones conocidas** coincidan con `text_normalized`; si la longitud no coincide, registrar fallo y error de longitud.
2. **Tasa de error en posiciones conocidas:** comparar exclusivamente posiciones conocidas **cuando exista alineación inequívoca** y registrar explícitamente las predicciones con inserciones/eliminaciones o longitud incompatible. Definir antes de publicar una política de alineación consistente para esos casos; no presentarla sin más como CER convencional.
3. **Diagnósticos:** informar por separado fallos de longitud, iluminación, baja resolución, perspectiva y otras condiciones cuando haya metadatos o criterios reproducibles.

El conjunto **ELPD/Kaggle (80 imágenes con transcripción íntegra auditada)** sirve para EM y CER convencionales; **UC3M-LP (491 imágenes en test con transcripción parcial)** se analizará con métricas parciales y **sus cifras no se compararán como si fueran la misma métrica**.

### 7.3. Condiciones de reproducibilidad

Para cada ejecución registrar: arquitectura y versión, URL/revisión y hash del checkpoint cuando sea posible, configuración de preprocesamiento, normalización del texto predicho, batch size, dispositivo, precisión numérica, tamaño de imagen de entrada, partición evaluada, versión de los scripts y resultados por muestra. **Los baselines deben comenzar sin fine-tuning**, usando pesos preentrenados para medir generalización inicial.

## 8. Limitaciones y cuestiones abiertas

1. **Anonimización:** todas las etiquetas son parciales; no hay referencia completa para evaluar caracteres tapados.
2. **Correspondencia JSON/JPG:** el emparejamiento por nombre se validó sistemáticamente por estructura/dimensiones y visualmente por muestras; `imagePath` permanece inconsistente en origen.
3. **Geometría:** los recortes son cajas envolventes con margen, no imágenes rectificadas; algunos incluyen fondo o una perspectiva importante.
4. **Calidad variable:** hay imágenes con altura mínima de 20 px y escenas oscuras, borrosas o inclinadas. No se excluyeron por estas características.
5. **Inspección visual no exhaustiva:** 60 muestras no garantizan ausencia de anomalías visuales en toda la colección.
6. **Independencia de particiones:** no consta una auditoría de identidad de vehículo/matrícula entre `train` y `test`; no asumirla al entrenar.
7. **Licencia:** verificar los términos originales antes de redistribución o usos no contemplados.
8. **Métrica parcial:** faltan definir/implementar las reglas para predicciones de longitud diferente y cualquier alineación con inserciones/eliminaciones.

## 9. Archivos de referencia y reproducibilidad

| Etapa | Script o resultado |
|---|---|
| Inspección del ZIP | `scripts/uc3m/inspect_uc3m_zip.py` |
| Diagnóstico de anotaciones y correspondencias | `scripts/uc3m/audit_uc3m_annotation_mapping.py`, `audit_uc3m_image_pairing.py`, `audit_uc3m_same_name.py` |
| Análisis de contornos no cuadriláteros | `scripts/uc3m/diagnose_uc3m_polygons.py`, `review_uc3m_variable_polygons.py` |
| Prueba visual de recortes | `scripts/uc3m/preview_uc3m_lpo_crops.py` |
| Generación definitiva | **`scripts/uc3m/prepare_uc3m_lpo.py`** |
| Auditoría final | **`scripts/uc3m/audit_uc3m_lpo_final.py`** |
| Resumen de preparación | `results/inspection/UC3M-LP/lpo_preparation/lpo_preparation_summary.txt` |
| Resumen de auditoría | `results/inspection/UC3M-LP/lpo_final_audit/audit_summary.txt` |
| Evidencias visuales | Las tres hojas de contacto y `contact_sheet_index.csv` en `lpo_final_audit/` |
| Inventario de datos procesados | **`processed/UC3M-LP_LPO/manifest.csv`** |

**Precaución:** la generación omite reescribir PNG ya existentes si su formato y dimensiones son válidos; para una regeneración estricta desde cero, realizarla en una carpeta de salida nueva y comparar resultados. No eliminar la versión auditada sin respaldo.

## 10. Conclusión

**UC3M-LP_LPO queda preparado y auditado para las siguientes fases de investigación.** Se dispone de **2.547 recortes**, con **491 imágenes de test**, trazabilidad hasta los archivos fuente, polígonos de 4/5/6 vértices conservados, **0 incidencias técnicas** y revisión visual muestral favorable. El dataset se incorpora al protocolo de **evaluación parcial de OCR**, manteniendo las posiciones anonimizadas y diferenciando explícitamente sus métricas de las de datasets con transcripciones completas.
