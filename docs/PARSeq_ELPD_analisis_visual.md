# PARSeq — Auditoría visual de errores sobre ELPD/Kaggle

**Modelo:** PARSeq preentrenado (zero-shot; sin fine-tuning). **Evaluación:** 80 imágenes del subconjunto test auditado de ELPD. **Fuente:** `predictions.csv` y tres hojas `contact_sheet_errors_01.jpg`, `_02.jpg`, `_03.jpg`. **Inspección visual:** 25/25 errores del baseline.

## 1. Métricas de referencia

- **Exact Match (EM):** 55/80 = **68,75 %**; 25 imágenes con al menos un error.
- **CER micro:** 38/556 = **6,83 %**.
- **Errores por imagen:** 16 casos solo inserciones, 3 solo sustituciones, 2 solo eliminaciones y 4 mixtos. **20/25 fallos contienen al menos una inserción**.
- **Checkpoint SHA-256:** `bb5792a68e367476abca029cbf8699abc805f3d3dc7e57aae45c8ec4f7b7cd00`.
- **Manifiesto SHA-256:** `c22ea5efd511ca82a3889ed08c2b79ae44c655bbfbddaa844f12d800169cfc1c`.

## 2. Hallazgos visuales principales

1. **Escudos, distintivos nacionales y códigos laterales:** aparecen junto a varios errores de inserción. Ejemplos evidentes: `ANAL334` (+B,+S), `DANKE82` (+S), `BL654BP` (+0), `CV194VA` (+2), `CY726SV` (+M). La correspondencia entre símbolo e inserción es una **hipótesis visual**; no se ha demostrado causalidad.
2. **Duplicación de letras/cifras:** `FR697HF` → `FR697HFHF`, `5888881` → `58888881`, `WW697BH` → `WWW697BH`, `Z3010AZ` → `ZZ3010AZ`.
3. **Ambigüedad entre caracteres:** `IOO8646` → `1008646` (I/1 y O/0), `P136KOG` → `PI36KOG` (1/I), `5B40001` → `5840001` (B/8).
4. **Baja nitidez/contraste:** `IICE3083`, `CM101LW`, `ESCC690`, las dos imágenes `TR95XMI`, `WW921MX`.
5. **Fallos de truncamiento:** `GB1ECB` → `GB` tiene cuatro eliminaciones a pesar de que los caracteres aparecen presentes en el recorte; `TR95XMI` pierde la I final en ambas vistas.

## 3. Observaciones de las 25 imágenes incorrectas

| Imagen | GT | OCR | D | Hallazgo visual / hipótesis |
|---|---|---|---:|---|
| `test__GB1ECB.png` | `GB1ECB` | `GB` | 4 | Placa austriaca con escudo interior; reconoce solo el prefijo GB aunque los demás símbolos se ven en la miniatura. Error severo no explicado solo por blur. |
| `test__IICE3083.png` | `11CE3083` | `EIICE30837` | 4 | Muy desenfocada y brillante; grafías 1/I ambiguas, posibles caracteres leídos en los márgenes. GT previamente corregido. |
| `test__I008646.png` | `IOO8646` | `1008646` | 3 | Placa griega; tres confusiones visuales I→1 y O→0, pese a confianza elevada; caso de ambigüedad alfanumérica. |
| `test__ANAL334.png` | `ANAL334` | `BANSAL334` | 2 | Escudos entre AN y AL y banda izquierda; añade B y S ajenos a la secuencia etiquetada. |
| `test__CM101LW.png` | `CM101LW` | `BCH101LW` | 2 | Imagen muy oscura y poco contrastada; segundo carácter parece H en miniatura, pero GT consigna M. Reauditar en resolución original, sin cambiar GT aún. |
| `test__FR697HF.jpg` | `FR697HF` | `FR697HFHF` | 2 | Placa francesa con bandas y distintivos; repite HF al final. No hay evidencia clara de que HF aparezca dos veces en la imagen. |
| `test__TR95XMI_1.png` | `TR95XMI` | `LTR95XM` | 2 | Recorte rosado con baja nitidez; inserta L al comienzo y omite I final (que resulta difícil de distinguir). |
| `test__WW921MX.jpg` | `WW921MX` | `WW921HX2` | 2 | Baja nitidez, caracteres finales difíciles: M→H y +2 final; posibles elementos laterales. |
| `test__5888881.png` | `5888881` | `58888881` | 1 | Secuencia de cifras 8 repetidas; una 8 extra, posiblemente por dificultades con separador/escudo. |
| `test__5B40001.png` | `5B40001` | `5840001` | 1 | B→8; tipografía de B similar a 8, junto a cifra/relieve de poco contraste. |
| `test__BL654BP.png` | `BL654BP` | `BL0654BP` | 1 | Escudo dentro de la placa entre letras y cifras; se añade 0. |
| `test__BZ310SW.jpg` | `BZ310SW` | `BZ310SWH` | 1 | Banda regional derecha, inclinación y resolución limitada; añade H al final. |
| `test__CN498JC.png` | `CN498JC` | `CN498JCB` | 1 | Distintivo lateral derecho con 92; añade B al final. |
| `test__CV194VA.jpg` | `CV194VA` | `CV194VA2` | 1 | Banda lateral derecha con 74; añade 2 al final. |
| `test__CY726SV.jpg` | `CY726SV` | `CY726SVM` | 1 | Distintivo lateral derecho con 82; añade M al final. |
| `test__DANKE82.png` | `DANKE82` | `DANSKE82` | 1 | Emblema entre DAN y KE; lo interpreta como S. |
| `test__ES484LJ.jpg` | `ES484LJ` | `BES484LJ` | 1 | Añade B antes de ES; posible confusión con la banda azul y borde izquierdo. |
| `test__ESCC690.png` | `ESCC690` | `LESCC690` | 1 | Imagen muy desenfocada; añade L antes de ES, posiblemente por borde/banda izquierda. |
| `test__MWWK908.png` | `MWWK908` | `MWWWK908` | 1 | Escudo entre letras; inserta W adicional en un grupo de W repetidas. |
| `test__PI36KOG.png` | `P136KOG` | `PI36KOG` | 1 | GT corregido P136KOG y predicción PI36KOG; trazo 1/I prácticamente idéntico en la imagen. No modificar GT sin nueva verificación. |
| `test__WANNE685.JPG` | `WANNE685` | `WANSNE685` | 1 | Escudo separa WAN y NE; añade S junto a esa separación. |
| `test__WW697BH.jpg` | `WW697BH` | `WWW697BH` | 1 | Repite W en el prefijo WW; distorsión y diseño lateral pueden influir. |
| `test__Z3010AZ.png` | `Z3010AZ` | `ZZ3010AZ` | 1 | Añade Z inicial antes de la Z real; posible duplicación de primer carácter. |
| `train__TR95XMI_2.png` | `TR95XMI` | `TR95XM` | 1 | Segunda imagen de la misma matrícula que TR95XMI_1; fuerte desenfoque, omite I final. |
| `train__VS236891.png` | `VS236891` | `VS2368910` | 1 | Añade 0 al final, cerca del escudo cantonal derecho. |

## 4. Etiquetas que requieren atención (sin modificación automática)

- **`test__CM101LW.png`:** en la miniatura, el segundo carácter puede parecer **H** en vez de **M**. Es una sospecha, no una corrección aprobada. Revisar el recorte original de máxima resolución y, si es posible, otras imágenes de la misma matrícula.
- **`test__PI36KOG.png`:** grafías **1/I** casi indistinguibles; GT previo `P136KOG` debe conservarse hasta una reauditoría independiente.
- **`test__I008646.png` y `test__IICE3083.png`:** GT previamente corregidos, los caracteres parecidos a dígitos/letras y la calidad exigen especial cautela.

**No cambiar el manifiesto ni recalcular las métricas del baseline únicamente por una apariencia ambigua en estas hojas reducidas.**

## 5. Recomendaciones experimentales

1. Mantener **las predicciones originales** y el protocolo exacto (EM/CER sin correcciones de país) para comparación con SVTRv2-T/S y MDiff4STR-S.
2. Para un análisis adicional, separar subgrupos con **escudos/distintivos**, **baja calidad**, **tipografías/alfabetos ambiguos** y **duplicaciones**. Las categorías se solapan.
3. Comparar por **imagen**, no solo por puntuación global, identificando fallos compartidos y recuperaciones entre modelos.
4. Si se prueba un posprocesador, declararlo **experimento separado**: no debe contaminar el baseline zero-shot.
5. Revisar el posible sesgo por **matrículas duplicadas en el conjunto evaluado**; los prefijos `train__` y `test__` son nombres heredados del dataset, no implican por sí solos una fuga en el subconjunto auditado.

## 6. Limitaciones del análisis

Las hojas de contacto muestran miniaturas y no permiten adjudicar con certeza todos los fallos. En particular, la coincidencia entre un elemento gráfico y una inserción no prueba que el modelo lo haya leído como ese carácter. La confianza aportada por el modelo no es una probabilidad calibrada de corrección. Esta auditoría cubre **las 25 imágenes fallidas**, no revisa las otras 55 predicciones correctas ni verifica por completo la calidad del GT.
