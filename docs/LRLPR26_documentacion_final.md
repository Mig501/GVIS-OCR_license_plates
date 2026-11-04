# LRLPR-26 — Preparación del conjunto HR para OCR

**Proyecto:** Evaluación y optimización de arquitecturas OCR para el reconocimiento de matrículas  
**Estado:** Preparación técnica y auditoría visual prioritaria finalizadas; apto para entrenamiento y validación interna, sujeto a las limitaciones descritas.  
**Últimas comprobaciones:** 4 de noviembre de 2026 

## 1. Procedencia y objetivo

- **Dataset:** LRLPR-26.
- **Repositorio de referencia:** https://github.com/raysonlaroca/lrlpr-26-dataset/
- **Acceso:** Descarga facilitada por los autores previa autorización. No publicar ni redistribuir el enlace privado, las imágenes ni las anotaciones sin comprobar los términos aplicables.
- **Uso en el proyecto:** Conjunto principal de entrenamiento y validación interna de arquitecturas de reconocimiento de texto en matrículas (OCR), sin entrenar el detector de matrículas.
- **Selección:** Se usan únicamente las imágenes de alta resolución (HR) de entrenamiento de los escenarios A y B; se excluyen las imágenes LR y el test oficial LR de esta preparación.

## 2. Composición del conjunto seleccionado

| Escenario | Disposición | Tracks | Imágenes HR |
|---|---|---:|---:|
| A | Brazilian | 5.000 | 25.000 |
| A | Mercosur | 5.000 | 25.000 |
| B | Brazilian | 2.000 | 10.000 |
| B | Mercosur | 8.000 | 40.000 |
| **Total** | | **20.000** | **100.000** |

Cada track contiene cinco imágenes HR y un fichero JSON con al menos los campos `plate_layout`, `plate_text` y `corners`. Las anotaciones del escenario A incluyen coordenadas de esquinas por imagen; en las muestras y la auditoría del escenario B, `corners` está vacío. Las imágenes HR de ambos escenarios ya contienen una matrícula localizada, por lo que se conservaron **sin rectificación geométrica**. La posible rectificación del escenario A queda reservada para una ablación posterior.

La auditoría de las anotaciones identificó **16.946 textos de matrícula distintos** y **1.646 etiquetas presentes en más de un track**. La coincidencia de texto no prueba identidad física del vehículo.

## 3. Manifiestos y particiones

El manifiesto original contiene una fila por imagen HR; las rutas son relativas a la carpeta del manifiesto. Incluye texto normalizado, escenario, tipo de matrícula, identificador de track, dimensiones y disponibilidad de esquinas, sin copiar las imágenes. Las etiquetas se conservan como `annotation_unverified`.

Las particiones se generaron con **semilla 42** y agrupación por texto de matrícula normalizado, manteniendo también las cinco imágenes de cada track en la misma partición:

| Partición | Imágenes | Tracks |
|---|---:|---:|
| Entrenamiento | 90.000 | 18.000 |
| Validación | 10.000 | 2.000 |
| **Total** | **100.000** | **20.000** |

**Solapamientos exactos train/val:** 0 textos de matrícula y 0 tracks.

| Escenario / disposición | Train (imágenes) | Val (imágenes) |
|---|---:|---:|
| A / Brazilian | 22.430 | 2.570 |
| A / Mercosur | 22.630 | 2.370 |
| B / Brazilian | 9.120 | 880 |
| B / Mercosur | 35.820 | 4.180 |

El manifiesto original no se modificó durante la creación de particiones.

## 4. Auditoría final de integridad

| Comprobación | Resultado |
|---|---:|
| Imágenes del manifiesto | 100.000 |
| Imágenes abiertas/verificadas | 100.000 |
| Incidencias técnicas detectadas | 0 |
| Anotaciones JSON revisadas | 20.000 |
| Tracks | 20.000 |
| Matrículas normalizadas distintas | 16.946 |
| Textos compartidos train/val | 0 |
| Tracks compartidos train/val | 0 |
| SHA-256 del manifiesto original | Coincide / intacto |
| Tiempo registrado para la auditoría optimizada | 2,4 minutos |

**Estado reportado por el script:** `APTO PARA ENTRENAMIENTO/VALIDACION`.

## 5. Auditoría visual de recortes LPO

Se realizó una **auditoría geométrica automática de las 100.000 imágenes HR** mediante `scripts/lrlpr/audit_lrlpr_lpo_visual.py`, complementada con hojas de contacto aleatorias y la revisión visual de todas las imágenes marcadas. El propósito fue detectar recortes que contuvieran vehículos completos u otras escenas en lugar de la matrícula localizada, como ocurrió en un dataset europeo previo.

| Grupo | Imágenes analizadas automáticamente | Alertas geométricas |
|---|---:|---:|
| Scenario-A / Brazilian | 25.000 | 30 |
| Scenario-A / Mercosur | 25.000 | 1 |
| Scenario-B / Brazilian | 10.000 | 0 |
| Scenario-B / Mercosur | 40.000 | 0 |
| **Total** | **100.000** | **31** |

Todas las alertas automáticas correspondían a **relación ancho/alto baja** (`aspecto_bajo`). No se detectaron imágenes imposibles de abrir en esta auditoría. Las hojas iniciales mostraban 20 de las 30 alertas de Scenario-A/Brazilian, además de la única alerta de Scenario-A/Mercosur. Las diez imágenes restantes se reunieron mediante `scripts/lrlpr/review_lrlpr_remaining_alerts.py` para completar la inspección visual de los casos señalados.

**Resultado de la revisión prioritaria:**

- **31/31 alertas geométricas revisadas visualmente**.
- **0 recortes incorrectos confirmados** en las imágenes revisadas.
- **0 imágenes excluidas o modificadas** tras esta auditoría.
- Las muestras aleatorias de los cuatro grupos mostraron matrículas localizadas, con variaciones de perspectiva, iluminación, resolución y márgenes; no se identificaron fotografías de vehículos completos en esas muestras.

**Alcance:** la comprobación de proporciones cubre las 100.000 imágenes y la inspección visual cubre las muestras generadas y las 31 alertas; **no equivale a una inspección visual individual de todas las imágenes**. Un fallo de recorte con geometría aparentemente normal podría no haberse detectado. No se aplicaron umbrales de exclusión basados únicamente en la proporción de imagen.

Informes conservados en `results/inspection/LRLPR-26/lpo_visual_audit/`: `visual_audit_summary.txt`, `geometry_alerts.csv` y hojas JPG de muestras/alertas. El dataset y sus particiones permanecieron intactos.

## 6. Reproducibilidad: scripts utilizados

Todos los scripts específicos se mantienen en `scripts/lrlpr/`:

1. `download_lrlpr.py` — descarga y comprobación CRC del ZIP.
2. `inspect_lrlpr_zip.py` — estructura interna y extensiones.
3. `extract_lrlpr_hr.py` — extracción selectiva de HR y JSON.
4. `inspect_lrlpr_annotations.py` — auditoría del esquema de anotaciones.
5. `inspect_lrlpr_hr_samples.py` — muestras visuales y comprobación estructural de tracks.
6. `build_lrlpr_hr_manifest.py` — manifiesto HR.
7. `split_lrlpr_hr_by_plate.py` — particiones agrupadas con semilla 42.
8. `check_lrlpr_split_integrity_fast.py` — auditoría final optimizada.
9. `audit_lrlpr_lpo_visual.py` — análisis geométrico de las 100.000 HR y hojas de contacto.
10. `review_lrlpr_remaining_alerts.py` — revisión visual de las diez alertas no mostradas en la primera hoja.

## 7. Limitaciones y precauciones experimentales

- **Auditoría visual no exhaustiva.** Las 31 alertas geométricas y las muestras aleatorias se revisaron, pero no se inspeccionaron manualmente las 100.000 imágenes; pueden subsistir errores de recorte no detectados.
- **Etiquetas sin auditoría humana exhaustiva.** La integridad técnica no demuestra la corrección visual del ground truth de las 100.000 imágenes.
- **Identidad de vehículo no demostrada.** Agrupar por texto evita coincidencias exactas, pero no garantiza ausencia total de fotografías del mismo vehículo con identificadores distintos, ni detecta duplicados visuales no exactos.
- **Diferencia entre escenarios.** A dispone de esquinas; B no. No se ha aplicado rectificación geométrica para mantener el protocolo de entrada inicial uniforme y reproducible.
- **Generalización geográfica.** Las disposiciones Brazilian y Mercosur no sustituyen una evaluación externa con matrículas españolas o europeas.
- **No es un test final independiente.** El conjunto `val` es de validación interna. El test oficial LR quedó fuera de la selección HR, y ELPD/otros datasets externos deberán mantenerse separados de las decisiones de entrenamiento y ajuste.
- **Licencia y privacidad.** Respetar las condiciones facilitadas por los autores. No redistribuir datos ni URLs de acceso privadas.

## 8. Uso recomendado

1. Utilizar `train_manifest.csv` para entrenamiento y `val_manifest.csv` para selección de configuraciones, manteniendo la semilla y agrupación documentadas.
2. Medir **Exact Match (EM)** y **Character Error Rate (CER)**; para comparar eficiencia, utilizar el mismo hardware y condiciones de inferencia.
3. Informar de resultados por escenario y formato, además de métricas globales.
4. Reservar ELPD verificado y otros datasets externos para generalización, sin utilizarlos para escoger hiperparámetros del entrenamiento.
5. Registrar versiones de código, checkpoints, hiperparámetros y decisiones de exclusión en cada experimento.

**Conclusión:** LRLPR-26 HR está técnicamente preparado para entrenamiento y validación interna de OCR sobre 100.000 imágenes, con manifiestos reproducibles, particiones agrupadas, una auditoría técnica sin incidencias y la revisión de las 31 alertas geométricas sin recortes erróneos confirmados. Esto no equivale a una validación humana exhaustiva de las imágenes o sus etiquetas.
