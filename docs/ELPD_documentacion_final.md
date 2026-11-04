# ELPD — Documentación de preparación y evaluación externa OCR

**Proyecto:** Prácticas GVIS — Evaluación de arquitecturas OCR para matrículas.  
**Dataset:** European License Plates Dataset (ELPD), Kaggle: `abdelhamidzakaria/european-license-plates-dataset`.  
**Fecha de cierre del subconjunto auditado:** 3 de noviembre de 2026.  
**Estado:** **subconjunto de evaluación externa APTO** (80 imágenes). **No** significa que se hayan verificado las 735 etiquetas.

## 1. Resumen ejecutivo

Se descargó y preparó ELPD como fuente de imágenes de matrículas europeas para probar la **generalización de los modelos OCR**, no como corpus principal de entrenamiento. Del total de **735 imágenes** se construyó un **test auditado de 80 imágenes, correspondientes a 73 matrículas distintas**, con transcripción comprobada visualmente: **76 etiquetas confirmadas sin cambios y 4 corregidas**.

La comprobación automática final sobre los archivos reales obtuvo **80/80 imágenes correctas**, **0 incidencias** y **0 grupos de imágenes idénticas por SHA-256** en el subconjunto. El manifiesto original y el ZIP fuente se conservaron. Las **655 imágenes fuera del test auditado no cuentan aún con verificación exhaustiva de ground truth**.

## 2. Procedencia, alcance y uso

| Campo | Descripción |
|---|---|
| Fuente | https://www.kaggle.com/datasets/abdelhamidzakaria/european-license-plates-dataset |
| Identificador | `abdelhamidzakaria/european-license-plates-dataset` |
| Formato encontrado | 735 archivos de imagen dentro de `dataset_final/{train,val,test}/`; sin archivos de anotación textual independientes |
| Etiqueta candidata | Derivada del nombre de archivo; normalización a mayúsculas y caracteres alfanuméricos, eliminando sufijos de copia cuando corresponde |
| Particiones originales | `train=588`, `val=74`, `test=73` |
| Uso en este proyecto | Evaluación externa europea de OCR sobre recortes LPO (*License Plate Only*) |
| Licencia y redistribución | **Pendiente de confirmar contra los términos vigentes en la fuente**. No suponer que la licencia del código o de otros repositorios se aplica a este dataset; no redistribuir sus imágenes sin autorización apropiada |

**Alcance de las métricas:** el test auditado permite calcular Exact Match (EM) y Character Error Rate (CER) a partir de `text_normalized`. La exactitud de las 655 transcripciones restantes no está certificada. Tampoco se ha comprobado que estas 73 matrículas no coincidan con otros datasets externos.

## 3. Preparación y auditoría realizadas

1. **Descarga e inventario.** Se descargó el ZIP de Kaggle, se comprobó su integridad y se inventariaron las 735 imágenes. Se construyó `ELPD_LPO/images/` y `manifest.csv` conservando, cuando ya eran recortes, los bytes originales.
2. **Auditoría automática de calidad.** Se señalaron **172/735 imágenes** con al menos una alerta heurística (oscuridad, contraste, desenfoque, tamaño, aspecto, etiquetas inusuales o cruce de particiones). Las alertas **no equivalen a errores ni exclusiones**.
3. **Revisión prioritaria.** Se examinaron 31 casos y se detectaron tres fotografías de escena completa que incumplían LPO. Se recortaron manualmente, se verificaron visualmente y se integraron preservando las escenas originales y un registro de integración.
4. **Revisión de solapamientos.** Se identificaron **15 identificadores presentes en varias particiones** (32 imágenes implicadas) y se compararon sus fotografías. Se generó una **propuesta** de agrupación por texto de matrícula, pero **no** se reescribieron las particiones originales.
5. **Revisión del ground truth.** Se inspeccionaron las 80 imágenes del `test` agrupado; 76 transcripciones se confirmaron y cuatro se corrigieron. Las ambiguas se comprobaron con los archivos originales antes del cierre.
6. **Cierre técnico.** Se construyó el manifiesto verificado, se comprobaron rutas, apertura de archivos, dimensiones, etiquetas, duplicados exactos y la invariancia del manifiesto general.

### 3.1. Tres escenas completas convertidas en LPO

| Archivo | Coordenadas del recorte (x1,y1,x2,y2) | Tamaño final |
|---|---|---|
| `test__PH20CNP.png` | `(676,392,785,424)` | `109 × 32` |
| `train__SB90TZI.png` | `(494,463,675,519)` | `181 × 56` |
| `train__TR05RBG.png` | `(710,439,944,515)` | `234 × 76` |

Se conservaron las condiciones originales de resolución, iluminación y ruido; no se aplicaron filtros de mejora a los recortes. Las **otras 732 imágenes** no se modificaron en esta fase.

### 3.2. Particiones originales y propuesta agrupada

| Partición | Original | Propuesta agrupada |
|---|---:|---:|
| Train | 588 | 571 |
| Val | 74 | 84 |
| Test | 73 | 80 |
| **Total** | **735** | **735** |

La propuesta reasigna **17 imágenes** (`train→test=7`, `train→val=10`) para que una misma transcripción candidata no esté en conjuntos distintos. **Es un manifiesto derivado**: `scenario` y el `manifest.csv` general continúan describiendo las particiones originales. Coincidencia de texto **no prueba identidad del vehículo**. El subconjunto final seleccionado consiste en las **73 imágenes del test original más 7 de train**, y debe tratarse en conjunto como **test externo**, sin usarse para ajuste de hiperparámetros ni entrenamiento.

## 4. Ground truth auditado

| Etiqueta del nombre original | Etiqueta final verificada |
|---|---|
| `PI36KOG` | `P136KOG` |
| `YB75711` | `YBZ5711` |
| `I008646` | `IOO8646` |
| `IICE3083` | `11CE3083` |

El CSV `scripts/elpd/gt_validation.csv` conserva `candidate_text`, `review_status` (`verified` o `corrected`), `verified_text` y notas. El archivo `gt_corrections.csv` enumera los cuatro cambios. El manifiesto de evaluación ya incorpora el texto corregido en `text_normalized`, preservando `text_candidate_original` y la ruta de imagen para auditoría.

**Distribución final del test:** 80 imágenes; 73 IDs de matrícula únicos tras corrección; 76 etiquetas verificadas sin cambios; 4 corregidas; 0 pendientes; 0 ilegibles. En el manifiesto final: `gt_review_status ∈ {verified, corrected}`, `strict_gt_eligible=yes` y `split_grouped=test` para todas las filas.

## 5. Comprobación final de integridad

| Prueba | Resultado |
|---|---:|
| Imágenes referenciadas | 80 |
| Imágenes válidas y legibles | 80 |
| Incidencias detectadas | 0 |
| Etiquetas verificadas / corregidas | 76 / 4 |
| Matrículas distintas | 73 |
| Filas del inventario general | 735 |
| Grupos de imágenes idénticas (SHA-256) en test | 0 |

**Resultado registrado:** `ESTADO: APTO PARA EVALUACION EXTERNA` (3 de noviembre de 2026). Los hashes corresponden a **archivos de manifiesto**, no constituyen por sí mismos una huella agregada de todo el dataset.

## 6. Cómo utilizarlo en los experimentos OCR

- **Fuente:** `results/audit/ELPD/final_eval/verified_test_manifest.csv`.
- **Raíz de imágenes:** `/home/miguel.sanchez/datasets/ocr_matriculas/processed/ELPD_LPO/`.
- **Entrada por ejemplo:** cargar `DATASET_ROOT / image_path`; utilizar `text_normalized` como etiqueta. Mantener `image_path` como ID de muestra y `plate_group_id` para métricas agrupadas por matrícula.
- **Preprocesamiento del modelo:** aplicar únicamente el redimensionamiento/normalización requerido por cada arquitectura, registrándolo. No alterar la imagen maestra ni su etiqueta.
- **Métricas:** Exact Match sobre la cadena completa y CER (distancia de edición acumulada / total de caracteres de referencia); reportar también número de muestras, país/condición cuando pueda verificarse y latencia bajo un hardware común.
- **Protocolo:** no usar este test para entrenamiento, selección de modelos o ajuste de hiperparámetros; reportar resultados como evaluación europea externa y no como precisión representativa de matrículas españolas.

Antes de publicar medidas conviene comprobar que no hay coincidencias o duplicados con los conjuntos que se usen para entrenamiento, incluidos otros datasets.

## 7. Limitaciones y tareas pendientes

1. **Cobertura:** solo 80/735 imágenes tienen GT plenamente auditado; las otras 655 mantienen etiquetas candidatas derivadas de nombres de archivo.
2. **Sesgo de selección:** el test procede de la partición `test` y de 7 ejemplos reagrupados, no de una muestra aleatoria estratificada de toda Europa; no inferir rendimiento poblacional general.
3. **Identidades:** la agrupación se hizo por cadena textual normalizada; no asegura identidad de vehículo ni evita por sí sola fugas respecto a otros datasets.
4. **Licencia y privacidad:** comprobar términos de Kaggle y derechos de reutilización antes de redistribuir archivos. Guardar los datos en los directorios controlados del clúster y no incorporar fotografías al repositorio público.
5. **Comparabilidad:** registrar versión del modelo, pesos, hardware, tamaño de lote y latencia en idénticas condiciones para SVTRv2, PARSeq, MDiff4STR y cualquier otro OCR.

**Criterio de cierre:** terminado el **subconjunto europeo de evaluación externa ELPD de 80 imágenes**. La validación exhaustiva de las 735 imágenes queda fuera de este alcance y requeriría trabajo adicional.

## 10. Fuentes y referencias

- Dataset de origen: https://www.kaggle.com/datasets/abdelhamidzakaria/european-license-plates-dataset
- Informes internos: `suspicious_images.csv`, `split_conflicts.csv`, `gt_validation.csv`, `verified_test_manifest.csv`, `integrity_summary.txt`.
- Identificadores de SLURM documentados durante el proceso: 70231 (auditoría), 70232 (prioridades), 70241 (recortes), 72194 (integración), 72200 (manifiestos derivados). El cierre final se acredita por `integrity_summary.txt`.
