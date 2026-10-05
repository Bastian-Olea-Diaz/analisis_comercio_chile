# Comercio exterior de Chile: análisis, grafos, forecasting, inferencia causal, experimentación y monitoreo de drift

Informe técnico del proyecto. Registra qué se hizo, por qué y qué alternativas se descartaron en cada etapa.
Los resultados numéricos provienen de los notebooks en `notebooks/`.

## Marco de trabajo: CRISP-DM

El proyecto se organizó según el ciclo CRISP-DM (*Cross-Industry Standard Process for Data Mining*). Cada
sección de este informe corresponde a una fase:

| Fase CRISP-DM | Secciones | Entregables |
|---|---|---|
| 1. Comprensión del negocio | §1 | Preguntas, objetivos analíticos y criterios de éxito |
| 2. Comprensión de los datos | §2, §3, §4 | `download.py`, `ingest.py`, `references.py`; `notebooks/01_eda` |
| 3. Preparación de los datos | §5, §6 | `clean.py`, `features.py`; panel producto × país × mes |
| 4. Modelado | §7, §8, §9 | `graph.py`, `forecast.py`, `classify.py`; `notebooks/02`–`04` |
| 5. Evaluación | §10 (técnica), §12 (frente a los objetivos) | Tests anti-leakage, baselines, `notebooks/05` |
| 6. Despliegue | §11, §13 | Monitoreo de drift (`drift.py`, `notebooks/06`), repositorio, informe y figuras |
| Segunda iteración del ciclo (pregunta causal) | §15 | `causal.py`, `tests/test_causal.py`; `notebooks/07` |
| Tercera iteración del ciclo (experimento A/B/n) | §16 | `experiment.py`, `tests/test_experiment.py`; `notebooks/08` |

CRISP-DM no es lineal: los hallazgos de una fase obligan a volver a otra. En este proyecto esas vueltas
cambiaron el alcance y el diseño de forma concreta:

| Vuelta | Qué la provocó | Qué cambió |
|---|---|---|
| Datos → Negocio | Solo 8 meses de datos (ene–ago 2026) | Se amplió el alcance a 2024–2026 para poder modelar estacionalidad y validar en el tiempo (§2.1) |
| Datos → Negocio | Los IDs de empresa se renumeran cada mes (§2.5) | Se descartaron las preguntas por empresa; la unidad de análisis pasó a ser producto × país |
| Datos ↔ Preparación | Defectos en los archivos crudos detectados al ingerir (§3.2–3.3) | Reglas de reparación en la ingesta y de imputación exacta en la limpieza (R2) |
| Modelado → Preparación | Un modelo global en escala absoluta "encogía" las series grandes (§9.1) | Features y objetivo relativos al nivel reciente |
| Evaluación → Preparación | Un test automatizado detectó leakage en la codificación de países (§6) | Ranking de socios calculado solo con el pasado; resultados recalculados |
| Evaluación → Modelado | L2 en log subestimaba 46%; el grafo aportaba solo con poca historia (§9.1, §10.2) | Pérdida L1 + Tweedie para totales; el grafo entró como feature del Problema 2 |
| Despliegue → Evaluación | El período de control del monitoreo generó falsas alarmas (§11.2) | Cinco correcciones al diseño del monitoreo antes de evaluar 2026 |
| Evaluación → Negocio | Los modelos predictivos no responden preguntas de impacto ("¿cuánto afectó el arancel?") | Nueva pregunta P5 y segunda iteración del ciclo con inferencia causal (§15) |
| Evaluación → Modelado | La triple diferencia no pasó su test de tendencias paralelas (§15.4) | La doble diferencia pasó a ser el diseño principal |
| Evaluación → Negocio | Medir un efecto pasado no dice qué intervención conviene hacer | Nueva pregunta P6: diseño y análisis de un experimento A/B/n (§16) |
| Evaluación → Diseño del experimento | Los tests A/A mostraron falsos positivos inflados con el análisis ingenuo y con revisiones mensuales (§16.3) | Errores agrupados por producto, SRM sobre la unidad aleatorizada y una sola revisión final |

---

## 1. Comprensión del negocio

*Fase CRISP-DM: comprensión del negocio.*

### 1.1 Problema
Chile es una economía muy abierta al comercio: el comercio de bienes equivale a más de la mitad de su PIB.
Sus exportaciones dependen de pocos productos (cobre, litio, fruta, salmón, celulosa) y de pocos mercados, y
sus importaciones, de combustibles, maquinaria y bienes de consumo. Esa estructura expone a la economía a los
precios internacionales y a la demanda de socios específicos. El Servicio Nacional de Aduanas publica cada
declaración de exportación (DUS) e importación (DIN) a nivel de ítem como datos abiertos, una fuente detallada
y poco explotada.

No hubo un cliente real. Las preguntas se plantearon desde la perspectiva de un **analista de comercio
exterior** o de una **agencia de promoción de exportaciones**: alguien que necesita entender qué está pasando,
anticipar los próximos meses y saber cuándo confiar en sus modelos.

### 1.2 Preguntas de negocio y objetivos analíticos

| Pregunta de negocio | Objetivo analítico | Dónde se responde |
|---|---|---|
| P1. ¿Qué explica la evolución reciente del comercio chileno y cuán concentrado está? | Descomponer el crecimiento en precio y volumen; medir concentración y estacionalidad | §4 |
| P2. ¿Qué productos comparten capacidades y hacia qué mercados puede diversificarse Chile? | Red de productos por capacidades compartidas; predecir nuevos pares producto–mercado | §7 |
| P3. ¿Cuánto se comerciará los próximos meses y qué flujos esporádicos se repetirán? | Regresión del valor a 1–3 meses por producto × país; clasificación de actividad de series intermitentes | §8–§10 |
| P4. ¿Cómo saber cuándo un modelo en producción deja de ser confiable? | Monitoreo de calidad de datos, drift y desempeño, con umbrales y acciones | §11 |
| P5. ¿Cuánto afectaron los aranceles de EE.UU. de 2025 a las exportaciones chilenas? (segunda iteración) | Estimar el efecto causal con diferencias en diferencias sobre un experimento natural | §15 |
| P6. ¿Qué tipo de apoyo convierte una oportunidad de exportación en exportación? (tercera iteración) | Diseñar, validar y analizar un experimento A/B/n semi-sintético con resultados reales del control | §16 |

Otra pregunta inicial, **el análisis por empresa** (comportamiento y continuidad de exportadores), se
descartó en la fase de comprensión de los datos: los identificadores anonimizados se renumeran cada mes (§2.5).

### 1.3 Criterios de éxito (fijados antes de evaluar)
- **Datos:** los totales reconstruidos deben coincidir con las cifras oficiales (Banco Central), sin pérdida de
  registros que no esté explicada.
- **Grafo:** solo se justifica si mejora, fuera de tiempo, la predicción de nuevos pares producto–mercado
  sobre un baseline sin grafo. Si no lo hace, se reporta como resultado negativo.
- **Modelos:** superar al mejor baseline simple en el período de test (ene–ago 2026), evaluado una sola vez.
  Durante la evaluación se agregó el requisito de respaldar la mejora con un intervalo de confianza.
- **Monitoreo:** sin falsas alarmas en un período de control y con detección del cambio conocido de 2026,
  usando umbrales definidos antes de mirar ese período.
- **Inferencia causal (P5, definido al diseñar la segunda iteración):** el diseño debe pasar un contraste de
  tendencias paralelas y una prueba placebo. Si el efecto no es significativo, se reporta como nulo junto con el
  efecto mínimo que el diseño podía detectar.
- **Experimento (P6, definido al diseñar la tercera iteración):** el procedimiento debe dar ~5% de falsos
  positivos en tests A/A con datos reales y recuperar sin sesgo el efecto inyectado.

### 1.4 Restricciones
Datos públicos y anonimizados; 32 meses de historia; sin precios internacionales de commodities; exportaciones
valoradas FOB e importaciones CIF (no son directamente comparables).

---

## 2. Datos

*Fase CRISP-DM: comprensión de los datos (descripción y verificación de la documentación).*

### 2.1 Fuentes

| Fuente | Período | Origen |
|---|---|---|
| Exportaciones (DUS) | ene 2024 – ago 2026 | datos.gob.cl: `registro-de-exportaciones-2024`, `-2025`, `registro-de-exportacion-2026` |
| Importaciones (DIN) | ene 2024 – ago 2026 | datos.gob.cl: `registro-de-importacion-2024`, `-2025`, `-2026` |
| Diccionario de datos DUS/DIN v2.0 | — | datos.gob.cl, `references/diccionario-aduana-dus-din-v2.0.xlsx` |
| Tablas de códigos (Anexo 51) | — | aduana.cl, Compendio de Normas Anexo 51 → `references/codigos/*.csv` |

**Por qué se agregaron 2024–2025.** El proyecto partió con ene–ago 2026, es decir, 8 observaciones mensuales agregadas: no alcanza
para entrenar ni validar honestamente un modelo, y no se puede aprender la estacionalidad anual (la fruta
concentra sus envíos entre noviembre y marzo). Con 32 meses es posible:
(a) modelar la estacionalidad comparando cada mes con el mismo mes del año anterior,
(b) tener varias ventanas de validación temporal y
(c) contar con una referencia histórica para el monitoreo de drift.

La descarga es reproducible (`python -m comercio_chile.download`) y queda registrada en
`data/raw/manifest.csv` (URL, tamaño y SHA-256 de los 93 archivos, ~1 GB comprimido).

### 2.2 Estructura y granularidad

- Archivos de texto separados por `;`, **sin encabezado**, codificación latin-1, un archivo por mes.
- **Una fila = un ítem de una declaración.** Una DUS/DIN puede tener muchos ítems (productos distintos).
- Exportaciones: 84 columnas (61 de encabezado + 23 de ítem). Importaciones: 178 columnas.
- Aduana publica también tablas de Bultos y Documentos de Transporte (se unen a la DUS por `NUMEROIDENT` +
  `FECHAACEPT`). Se revisaron en la exploración inicial, pero no aportan al análisis y no se procesan.

### 2.3 Hallazgos sobre la documentación

1. **Los `descripcion-y-estructura-de-datos.xlsx` publicados con los datasets 2026 de exportaciones e
   importaciones son idénticos** (mismo MD5) y solo describen la DIN (importaciones). La estructura de exportaciones no venía documentada. Se obtuvo el diccionario oficial
   v2.0 desde datos.gob.cl, que incluye DUS, Bultos y Documentos de Transporte. Su hoja `titulos` tiene el
   orden exacto de columnas, y se validó contra los datos (84 campos por línea; la columna 69 contiene códigos
   arancelarios válidos, como `08092919` = cerezas frescas).
2. **El diccionario es la única fuente de verdad del esquema.** `schema.columns()` lee los nombres desde el
   `.xlsx` en vez de copiarlos a mano.

### 2.4 Variables clave y una trampa de agregación

| Concepto | Exportaciones (DUS) | Importaciones (DIN) |
|---|---|---|
| Fecha de aceptación | `FECHAACEPT` | `FECACEP` |
| Id de declaración | `NUMEROIDENT` | `NUMENCRIPTADO` |
| Valor del **ítem** (US$) | `FOBUS` (FOB) | `CIF_ITEM` (CIF) |
| Producto | `CODIGOARANCEL` (SA 8 dígitos) | `ARANC_NAC` |
| País socio | `PAISDESTINO` | `PA_ORIG` (origen) |
| Empresa (anonimizada) | `NRO_EXPORTADOR` | `NUM_UNICO_IMPORTADOR` |

**Trampa de agregación:** las columnas de encabezado (`TOTALVALORFOB`, `VALORCIF`, `FOB`, `CIF`, `TOT_PESO`) se
**repiten en cada ítem**. Sumarlas cuenta varias veces las declaraciones con muchos ítems. Siempre se agregan
los valores **a nivel ítem**. Verificación: en la DIN 26416906 (6 ítems), la suma de `CIF_ITEM`
(99.424,93) coincide exactamente con su `CIF` de encabezado.

**FOB vs. CIF.** Las exportaciones se valoran FOB (el valor de la mercancía puesta en el barco) y las
importaciones CIF (FOB + flete + seguro). Por eso la balanza comercial calculada con estos datos no es
estrictamente comparable con la oficial del Banco Central, que usa FOB para ambos flujos.

### 2.5 Hallazgo crítico: los IDs de empresa se renumeran cada mes

El diccionario indica que el RUT del exportador/importador se entrega "como número único correlativo".
Se verificó si ese correlativo es estable en el tiempo, usando como control un atributo que debería ser fijo
para una empresa: su comuna.

| | Comuna única por ID **dentro** del mes | Comuna única por ID **entre** meses |
|---|---|---|
| Exportadores | 97,6 % | 1,1 % |
| Importadores | 98,6 % | 0,1 % |

*(32 meses, ene-2024 a ago-2026; ver `notebooks/01_eda.ipynb` §8)*

Además, en importaciones el ID máximo de cada mes ≈ número de importadores únicos de ese mes (correlativo 1…N).

**Conclusión:** el ID identifica a una empresa solo **dentro de un mes**. Implicancias:
- No hay series de tiempo por empresa ni grafos de empresas que evolucionen en el tiempo.
- Las dimensiones estables son los códigos oficiales: producto (SA), país, puerto, aduana y región.
- Las métricas por empresa son válidas mes a mes (por ejemplo, concentración de mercado).
- Se descarta de forma explícita intentar re-identificar empresas cruzando atributos: la anonimización es
  intencional y vulnerarla sería un problema ético.

### 2.6 Tablas de códigos

Las importaciones traen solo códigos (`PA_ORIG`, `ADU`, `VIA_TRAN`...). Se extrajeron las tablas oficiales del
Anexo 51 de Aduana (`python -m comercio_chile.references`): aduanas, tipos de operación, países, vías de
transporte, regiones y puertos. Los códigos se normalizan sin ceros a la izquierda (`03` → `3`), como en los datos.

**Pseudo-países.** Los códigos ≥ 900 y el 997 no son países socios: 901 combustibles para naves extranjeras,
902 "rancho" (provisiones para naves), 904 orígenes varios o no precisados, 905/907/910 zonas francas,
997 Chile (mercancía nacional que reingresa), 999 desconocido. Quedan marcados con `pseudo_pais = True`
para decidir explícitamente cómo tratarlos (en especial en el grafo, donde crearían nodos artificiales
muy conectados).

---

## 3. Ingesta y calidad de datos

*Fase CRISP-DM: comprensión de los datos (recolección inicial y verificación de calidad).*

### 3.1 Pipeline

```
download.py    ->  data/raw/{flujo}/{año}/*.rar|zip  +  manifest.csv
ingest.py      ->  data/interim/{flujo}/{AAAA-MM}.parquet  +  _ingest_log.csv
references.py  ->  references/codigos/*.csv
clean.py       ->  data/processed/panel_{flujo}.parquet
```

**Decisiones:**
- **Parquet (compresión zstd)** en vez de CSV: formato columnar, tipado y comprimido. 2026 pasó de ~3 GB de
  texto a ~340 MB, y la lectura es entre 10 y 50 veces más rápida.
- **Leer todo como texto y tipar después**, con `errors="coerce"`, **contando** cada valor no convertible.
  Así ningún dato se pierde en silencio por una inferencia automática de tipos.
- **Códigos como texto**, no como números: son etiquetas nominales (el país 225 no es "mayor" que el 220) y el
  código arancelario necesita sus ceros a la izquierda (el capítulo `03` = pescados).
- **Interim = copia fiel tipada.** No se filtra ni se agrega nada en esta capa; esas decisiones se toman
  después, de forma explícita.

### 3.2 Problemas encontrados en los archivos crudos y su solución

| Problema | Dónde | Causa | Solución |
|---|---|---|---|
| Registros partidos en 2 líneas | DIN may-2026 (6 registros) | Salto de línea dentro de `NUM_CONOC` | Si una línea tiene < 178 campos y unida a la siguiente suma exactamente 178, se unen |
| Filas partidas por `\r` | DUS ene y feb 2026 | Glosa de puerto `QASIM INTERNATIONAL\r` (Pakistán) trae un retorno de carro suelto | Un `\r` no seguido de `\n` se reemplaza por espacio |

Resultado en 2026: **0 filas descartadas**, 0 montos no convertibles, 0 fechas inválidas y 0 filas fuera del
mes del archivo. El detalle por archivo está en `data/interim/_ingest_log.csv`.

### 3.3 Ingesta histórica 2024–2025

La ingesta histórica encontró un patrón adicional: **registros DUS truncados en origen** (6 a 29 por mes en
2024 y en enero de 2025, 171 en total). Antes de aceptar el relleno se verificó que fuera seguro. Las líneas
tienen exactamente 63 campos (61 de encabezado + `NUMEROITEM` + `NOMBRE`): el corte ocurre **al final**, así
que los campos presentes conservan su posición y no hay desplazamiento de columnas. Algunas descripciones
contienen `&#X0D` (un retorno de carro codificado como entidad HTML), lo que sugiere la causa del corte en
el sistema de origen.

| Tipo | Ítems | Líneas unidas | Líneas rellenadas | Descartadas | Valores no convertibles |
|---|---|---|---|---|---|
| Exportaciones (DUS) | 3.152.344 | 0 | 171 | 0 | 0 |
| Importaciones (DIN) | 12.456.918 | 9 | 38 | 0 | 0 |

Los 15,6 millones de ítems ocupan 1,35 GB en Parquet.

---

## 4. Análisis exploratorio (EDA)

*Fase CRISP-DM: comprensión de los datos (exploración). Responde la pregunta P1.*

Notebook: `notebooks/01_eda.ipynb` · Figuras: `reports/figures/01_*.png` a `08_*.png`.

### 4.1 Validación de totales
Exportaciones 2024 = **US$ 99.629 millones** (todas las operaciones), consistente con la cifra anual del
Banco Central (≈ US$ 100 mil millones). Calzar con una fuente independiente valida la ingesta y la regla
de agregación a nivel ítem. Sumar el encabezado habría inflado el total ×1,8.

| Año | Exportaciones FOB (todas) | Exportaciones de bienes | Importaciones CIF |
|---|---|---|---|
| 2024 | 99.629 | 96.196 | 78.446 |
| 2025 | 107.027 | 103.121 | 85.672 |
| 2026 (ene–ago) | 89.222 | 85.454 | 58.616 |

### 4.2 Segmentos de la DUS
96,2% bienes, 3,0% servicios (tipo 202) y 0,8% rancho (tipo 211: combustible y provisiones a naves
extranjeras). El ítem más grande de toda la base (US$ 484 millones, ene-2026) es combustible para naves.
Sin separar segmentos, un solo registro distorsiona el mes completo.

### 4.3 El crecimiento de 2026 es de precios, no de volumen
Las exportaciones de bienes crecen **+26,3%** en ene–ago 2026 (las importaciones, +4,5%). Minerales
metalíferos (+8.787 MMUSD), cobre (+3.909), química inorgánica/litio (+3.311, +106%) y metales preciosos
(+2.062, +112%) explican casi todo el aumento. Fruta (−11%), celulosa (−8%), madera (−13%) y vino (−6%)
caen. En cátodos de cobre (HS 740311), el **valor unitario sube ~50%** sobre el promedio 2024, mientras el
**volumen cae ~20%**.

**Implicancias:** (1) el precio de los commodities es exógeno y no está en los datos, por lo que un modelo
entrenado con 2024–2025 tenderá a subestimar 2026; (2) 2026 es un cambio de régimen real que sirve para
validar el sistema de drift; (3) +26% nominal no es +26% de mercancía.

### 4.4 Composición y concentración
| | Exportaciones | Importaciones |
|---|---|---|
| Capítulo principal | Minerales metalíferos 37% + cobre 20% | Combustibles 17%, maquinaria 13% |
| Socio principal | China 37%, EE.UU. 16%, Japón 9% | China 26%, EE.UU. 19%, Brasil 9%, Argentina 9% |
| Top 10 productos HS6 | 67% del valor | 24% del valor |
| Productos para 80% / 95% | 27 / 132 | 443 / 1.334 |

Empresas exportadoras (dentro del mes): las 10 mayores generan ~48% del valor, pero el **HHI es bajo
(450–580)** porque ninguna domina sola y hay una cola de ~2.900 empresas. Se reportan ambas métricas porque
describen aspectos distintos de la concentración.

### 4.5 Estacionalidad: dos regímenes
Correlación de cada capítulo con su valor 1 mes antes (lag 1) y 12 meses antes (lag 12):
- **Estacionales** (lag 12 > lag 1): fruta (0,9 vs 0,4), vino (0,6 vs 0,0), pescados, carne y conservas.
- **Persistentes** (lag 1 > lag 12): química inorgánica/litio (0,8 vs −0,3), metales preciosos, minerales.
- **Ruidosas** (ambas bajas): cobre refinado, celulosa, madera.

**Implicancia:** todo modelo debe compararse contra el baseline *naive* (mes anterior) **y** el *naive
estacional* (mismo mes del año anterior).

### 4.6 Intermitencia
De 36.800 series producto(HS6)×país en exportaciones, **57% tiene comercio en 3 meses o menos** (0,7% del
valor), mientras que el 2,8% que comercia los 32 meses concentra **72% del valor**. En importaciones, el patrón
es el mismo (7,8% de las series = 75% del valor). Esto da lugar a dos problemas predictivos distintos:
regresión en las series regulares y clasificación de actividad en las esporádicas.

### 4.7 Distribución de valores
Los valores por ítem son aproximadamente log-normales (mediana ~US$ 11 mil en exportaciones y ~2 mil en
importaciones; máximo de cientos de millones). Se modelará en `log1p`, y los extremos **no** se eliminan
como outliers, porque son embarques reales.

### 4.8 Calidad
Sin duplicados; > 99,9% de las declaraciones con Σ ítems = encabezado; 71 ítems con valor negativo
(ajustes y rectificaciones, −0,16 MMUSD); 171 + 29 ítems truncados.

## 5. Limpieza y preparación

*Fase CRISP-DM: preparación de los datos (selección, limpieza, integración y formato).*

Código: `src/comercio_chile/clean.py` → `data/processed/panel_{exportaciones,importaciones}.parquet`.

**Unidad de análisis:** panel mensual `periodo × segmento × HS6 × país`, con `valor_usd`, `cantidad`,
`peso_kg` (solo DUS), `n_items`, `n_decl` y `n_empresas`. Exportaciones: 251.200 filas; importaciones: 878.710.

**Por qué HS6:** es el estándar internacional, comparable entre países. Los dígitos 7–8 son aperturas
nacionales que cambian con cada actualización del arancel chileno y romperían la continuidad de las series.

| Regla | Qué hace | Por qué |
|---|---|---|
| R1 Segmento | DUS: bienes (200, 210), servicios (202), rancho (211) | Servicios y rancho no usan el SA y distorsionan el análisis de bienes |
| R2 Imputación de truncados | Valor del ítem = total del encabezado − Σ resto, si es el único nulo de la declaración | Imputación exacta, no estadística. Recupera 156 + 13 ítems (2,3 + 0,26 MMUSD); el producto queda `000000` |
| R3 Negativos | Se conservan | Son ajustes legítimos que netean los totales |
| R4 Pseudo-países | Se conservan, con `pseudo_pais = True` | Se excluyen de forma explícita donde distorsionan (socios, grafo) |
| R5 Empresas | `n_empresas` se cuenta dentro del mes | Los IDs se renumeran cada mes |

## 6. Ingeniería de características

*Fase CRISP-DM: preparación de los datos (construcción de atributos).*

Código: `src/comercio_chile/features.py`. Se construye una **matriz densa serie × mes** (serie = flujo ×
HS6 × país; 0 = mes sin comercio): 36.772 series de exportación y 86.200 de importación × 32 meses. Así
todas las features son operaciones vectorizadas sobre columnas ≤ t, y la regla anti-leakage es verificable.

| Grupo | Features | Motivación (EDA) |
|---|---|---|
| Rezagos | `lag_0..5` (log), `lag_12` | Persistencia (minería, litio) |
| Estacionalidad | `lag_estacional` = mismo mes del año anterior al **objetivo**; `desvio_estacional` | Fruta, vino, salmón |
| Nivel y ventanas | medias de 3/6/12 meses (log y log de la media), `std_log_12`, `meses_activos_12`, `meses_desde_actividad` | Nivel, volatilidad, regularidad |
| **Relativas (sin escala)** | `rel_*` = feature − log(1 + media 3m) | Permiten que un modelo global trate igual a series grandes y pequeñas |
| Crecimiento | `crec_interanual_3m`, `cap_crec_interanual`, `cap_mom_3m` (capítulo completo) | Momentum de la serie y del sector |
| Precio | `precio_vs_mediana12` (valor unitario actual vs mediana de 12 meses) | El shock de 2026 fue de precios |
| Actividad | `log_n_decl`, `log_n_emp_media3` (empresas contadas dentro del mes) | Base de exportadores/importadores |
| Categóricas | `capitulo`, `pais` (top 40 socios **calculado con meses ≤ t**), `mes_objetivo`, `commodity` | Patrones compartidos entre series |
| Solo Problema 2 | `activo_t`, `activo_t1/t2`, `activo_estacional`, `activos_3/6` + **grafo**: `densidad`, `ubicuidad_p`, `diversidad_c` (grafo reconstruido en cada origen con los 12 meses previos) | Patrón de actividad; relatedness |

**Control de calidad de features:** antes de entrenar se revisaron las estadísticas descriptivas de cada
feature. Eso detectó un bug: `meses_desde_actividad` tomaba valores 9–11 cuando debía ser 0 para series
activas (calculaba la posición del último mes activo en la ventana, no la distancia a t). Un modelo con ese
error habría entrenado sin quejarse.

**Tests anti-leakage** (`tests/test_no_leakage.py`, 6 tests): alteran con ruido todos los datos posteriores
al origen t₀ y verifican que las features, la selección de series y el set de entrenamiento queden
idénticos. **El test encontró un leakage real:** el ranking de los 40 países principales (para la categoría
`pais`) se calculaba con todo el período, incluidos los meses futuros. Se corrigió, y todos los resultados
reportados usan la versión corregida.

## 7. Construcción y análisis del grafo

*Fase CRISP-DM: modelado descriptivo (§7.1–7.3) y su evaluación predictiva (§7.4). Responde la pregunta P2.*

Notebook: `notebooks/02_grafo.ipynb` · Código: `src/comercio_chile/graph.py`, `activation.py` · Figuras `09`–`11`.

### 7.1 Por qué este grafo
Un grafo se justifica cuando importan las conexiones indirectas. Se descartaron: la red Chile→países (una
estrella, equivalente a una tabla), la red de empresas en el tiempo (IDs renumerados), el *Product Space*
clásico (requiere datos de todos los países) y el vínculo importador–exportador (numeraciones
independientes). Se eligió una **red de capacidades**: dos productos HS6 se relacionan si **las mismas
empresas** los exportan. La co-ocurrencia se cuenta **por empresa-mes** y luego se suma, así que nunca se
vincula una empresa entre meses.

### 7.2 Construcción y decisiones
- **Ventana:** 2025 (12 meses), exportaciones de bienes.
- **Umbral de US$ 5.000 por empresa-mes-producto:** elimina ~40% de los pares (muestras, repuestos), pero
  solo ~0,05% del valor.
- **Ponderación de Newman (1/(k−1)):** el 0,9% de las empresas-mes (distribuidores con > 10 productos)
  generaba el 54% de los pares co-ocurrentes. Cada empresa reparte "una unidad" de vínculo entre sus productos.
- **Proximidad:** φ(p,q) = Σ_f w_f X_fp X_fq / max(N_p, N_q), con N_p ≥ 10 empresas-mes.
- **Resultado:** 1.137 productos (99,3% del valor), 21.008 aristas, densidad 3,4%, grado mediano 21.

### 7.3 Comunidades (Louvain)
Resolución 1,0 (barrido 0,5–2,0): **modularidad 0,757**, **estabilidad ARI 0,90** entre 10 semillas, 30
comunidades (19 con ≥ 10 productos); 85% del peso de las aristas queda dentro de las comunidades. **NMI ≈
0,5 contra la clasificación arancelaria:** las comunidades se parecen a ella pero no son iguales, y esa
diferencia es la información que aporta el grafo.

| Comunidad | % valor 2025 | Contenido destacado | % a China |
|---|---|---|---|
| Gran minería | 58,5 | Concentrado y cátodos de cobre **+ oro** (subproducto) | 47 |
| Fruticultura fresca | 9,6 | Cerezas, uva, manzanas | 36 |
| Acuicultura | 8,7 | Salmón atlántico y coho | 7 |
| Forestal: celulosa y papel | 3,5 | Celulosa **+ cartulina** (integración vertical) | 58 |
| Salares | 3,2 | **Carbonato de litio + nitrato de potasio** (mismos salares) | 48 |
| Química | 2,2 | Yodo, metanol | 34 |
| Forestal: madera | 2,1 | Madera aserrada, plywood (separada de la celulosa) | 5 |
| Manufactura diversificada | 1,4 | 265 productos (maquinaria, repuestos), hacia Perú y Bolivia | 1 |

**Productos puente** (coeficiente de participación de Guimerà & Amaral): aceite de oliva, fungicidas,
**alimento para peces** (agroindustria ↔ acuicultura), **cajas plásticas** (packaging compartido), **cable
de cobre** (minería ↔ manufactura). Son eslabones de cadenas de suministro entre ecosistemas.

**Riesgo por ecosistema:** minería, celulosa y salares (> 65% del valor) envían ~50% o más a China; la
acuicultura, la madera y la agroindustria están diversificadas.

**Red bipartita ecosistema–mercado** (figura `12`). Los productos se agregan en sus 8 ecosistemas principales
y se conectan con los 12 mercados principales, con aristas ponderadas por valor. Una red producto–país
completa (1.137 × ~180) sería ilegible. Métrica por mercado: **número efectivo de ecosistemas** = 1/HHI
de las participaciones de cada ecosistema en sus compras.

| Mercado | N.º efectivo de ecosistemas | Ecosistema principal |
|---|---|---|
| Suiza | 1,0 | Gran minería, 98% (oro para refinerías) |
| India | 1,4 | Gran minería, 84% |
| China / Japón | 1,8 | Gran minería, 73% / 72% |
| EE.UU. / Brasil | 3,2 | Gran minería, ~50% (+ salmón, fruta, madera) |
| México | 6,7 | El más diversificado |

Implicancia: la diversificación de una relación comercial depende de **cuántos ecosistemas** la
sostienen, no solo de cuántos países compran. Es un resultado descriptivo, no validado como predictivo.

### 7.4 Validación predictiva del grafo (fuera de tiempo)
**Hipótesis:** si el grafo captura capacidades reales, la **densidad de relatedness**
ω(p,c) = Σ_q φ(p,q)·M(q,c) / Σ_q φ(p,q) debería anticipar qué pares (producto, país) nuevos empieza a
exportar Chile.

**Protocolo:** validación = grafo 2024 → activaciones ene–ago 2025 (elegir variante y umbrales); test =
grafo 2025 → activaciones ene–ago 2026 (**evaluación única** con configuración congelada). Candidatos: pares
con < US$ 10 mil en la ventana base (~190 mil); activación: ≥ US$ 10 mil después (~1%). Baseline sin grafo:
ubicuidad del producto, diversidad del país y tamaños. Modelo: gradient boosting. CV con *GroupKFold* por
producto.

| Test (2025 → 2026) | AUC | AP | Precisión@100 | Precisión@1000 |
|---|---|---|---|---|
| Baseline sin grafo | 0,904 | 0,098 | 25% | 17,4% |
| **Baseline + grafo** | **0,914** | **0,115** | **31%** | **20,6%** |
| Densidad sola | 0,867 | 0,075 | 6% | 9,7% |
| Azar | 0,5 | 0,0097 | ~1% | ~1% |

- ΔAP en test **+0,017, IC 95% [0,011; 0,024]** (bootstrap por producto), igual que en validación (+0,0165).
- **Robusto:** con ponderación sí/no la diferencia es despreciable; en 9 combinaciones de umbrales, mejora
  de +13% a +17% en 8 de ellas (+5% en la restante); el baseline no lineal no absorbe la mejora.
- **Negocio:** de los 1.000 pares mejor rankeados, ~21% se activó en 2026 (lift ~20× sobre el azar). La
  mayoría de los aciertos son manufacturas hacia Latinoamérica.
- **La densidad sola rankea mal el top** (encuentra pares plausibles en mercados pequeños): el grafo
  **complementa** a las variables simples, no las reemplaza.

**Uso posterior:** la densidad y la comunidad entran como features del Problema 2 (actividad), recalculadas
en cada origen temporal solo con datos pasados.

**Limitaciones:** solo exportaciones de Chile; etiquetas de producto derivadas de texto libre; comunidades
sensibles a los umbrales (los resultados predictivos son robustos a ellos).

## 8. Definición de los problemas predictivos

*Fase CRISP-DM: modelado (selección de técnicas y diseño de la evaluación). Responde la pregunta P3.*

El EDA mostró que las series producto–país son de dos tipos: **regulares** (comercio casi todos los meses;
~75% del valor) e **intermitentes** (comercio esporádico). Son dos preguntas distintas:

| | Problema 1 · Regresión | Problema 2 · Clasificación |
|---|---|---|
| Pregunta | ¿Cuánto se comerciará? | ¿Habrá comercio? |
| Población (en cada origen, con datos pasados) | Series con comercio en ≥ 10 de los últimos 12 meses (~2.850 expo, ~13.700 impo; ~87–90% del valor del mes siguiente) | Series con comercio en 1–9 de los últimos 12 meses (~22 mil expo, ~51 mil impo) |
| Objetivo | Valor US$ en t + h, h = 1, 2, 3 | 1 si valor en t + 1 > 0 (tasa base 20–24%) |
| Métricas | WAPE (US$), MAE log, sesgo del total | AP, AUC, Brier, log-loss, calibración |

**Protocolo temporal común (walk-forward):** para el mes objetivo T, el origen es t₀ = T − h. Se entrena
solo con pares cuyo objetivo ya se observó (t + h ≤ t₀) y se reentrena cada mes. **Validación:** objetivos
jul–dic 2025 (selección de configuración). **Test:** objetivos ene–ago 2026, con evaluación **única** y
configuración congelada.

## 9. Modelos y entrenamiento

*Fase CRISP-DM: modelado (construcción y evaluación técnica de los modelos).*

**Modelo:** LightGBM **global**, uno por flujo (y por horizonte en el P1), entrenado con todas las series a la
vez. Un modelo por serie (ARIMA) tendría ~20 observaciones por serie; un modelo global aprende patrones
compartidos desde miles de series (enfoque ganador de la competencia M5).

### 9.1 Problema 1: la formulación importó más que los hiperparámetros

| Intento (validación, h=1) | Expo WAPE | Impo WAPE | Lección |
|---|---|---|---|
| A · L2 sobre log(1+y) | 0,590 | 0,554 | L2 estima la **media del log**: los ceros la arrastran hacia abajo y `exp` da la media geométrica (Jensen). Sesgo −47% |
| B · L1 sobre log(1+y) | 0,556 | 0,505 | L1 estima la **mediana**, que conmuta con log: sin sesgo de retransformación, óptima para error absoluto. Mejora el error por serie, pero el WAPE sigue mal |
| **C · L1 sobre cambio relativo** | **0,311** | **0,419** | El problema era la **escala**: los árboles agrupaban series gigantes con pequeñas. El objetivo log(1+y_{t+h}) − log(1+media 3m) es **sin escala** |
| E · Tweedie (US$) con offset log(nivel) | 0,310 | 0,432 | Modela la **media**: mejor total agregado (sesgo −3%/−4%), peor error por serie |
| Baseline media móvil 3 | 0,316 | 0,442 | — |

**Sensibilidad:** ponderar por valor (log, √) y la complejidad del árbol (15/31/63 hojas) cambian el WAPE en
±0,01, sin un patrón consistente entre flujos. **Decisión por parsimonia:** la configuración más simple (sin
ponderar; 31 hojas; 500 árboles; tasa 0,03). Elegir la "mejor" de 11 variantes con 6 meses de validación
sería sobreajustar al período de validación.

**Configuración final:** dos modelos según el uso. **L1 sobre cambio relativo** (mediana) para pronósticos
por serie y **Tweedie con offset** (media) para totales agregados.

### 9.2 Diagnóstico de sobreajuste/subajuste (P1, origen sep-2025)
- **Curva de aprendizaje:** el error de validación alcanza su mínimo cerca de los ~180 árboles y luego sube
  levemente (+0,3% a 500; +0,9% a 1.500): **sobreajuste leve**. No se ajustó con un solo origen; queda como
  mejora posible (*early stopping* interno en cada origen).
- **Curva de complejidad:** subajuste con 2 hojas (error de entrenamiento alto); validación plana entre 4 y
  63 hojas; **sobreajuste claro desde 127 hojas** (la validación sube y la brecha crece de forma monótona).

### 9.3 Problema 2
LightGBM binario (log-loss), con las mismas features más el patrón de actividad y el grafo. Ablación con y
sin las features del grafo. El grafo de importaciones se construye con importadores (productos que compran
las mismas empresas).

## 10. Evaluación y comparación

*Fase CRISP-DM: evaluación técnica de los modelos en el período de test. La evaluación frente a los objetivos de negocio está en §12.*

### 10.1 Problema 1 · Test ene–ago 2026 (WAPE)

| | h = 1 | h = 2 | h = 3 |
|---|---|---|---|
| **Expo · L1** | **0,319** | **0,354** | **0,362** |
| Expo · Tweedie | 0,319 | 0,367 | 0,379 |
| Expo · media móvil 3 (mejor baseline) | 0,362 | 0,401 | 0,419 |
| Expo · naive / naive estacional | 0,391 / 0,437 | 0,430 / 0,436 | 0,449 / 0,433 |
| **Impo · L1** | **0,409** | **0,427** | **0,443** |
| Impo · Tweedie | 0,426 | 0,450 | 0,470 |
| Impo · media móvil 3 | 0,459 | 0,482 | 0,506 |

- **L1 supera a todos los baselines en los 3 horizontes y en ambos flujos** (WAPE −11% a −14% vs la media
  móvil), **en los 8 meses** de 2026, con ΔWAPE = −0,046 (IC 95% [−0,104; −0,014]) en exportaciones y
  −0,050 ([−0,062; −0,038]) en importaciones (bootstrap por serie).
- **Por segmento (h=1):** en exportaciones, la ganancia está en los **no-commodities** (0,396 vs 0,548); en
  **commodities** apenas (0,294 vs 0,300), porque su precio es exógeno.
- **Validación vs test:** en exportaciones, L1 empataba con la media móvil en validación, pero la supera en el
  test 2026, cuando los precios subieron rápido y la media móvil reaccionó con rezago.
- **Sesgo del total:** L1 subestima (−9% a −13% en expo 2026; ~−10% en impo), como predice la teoría
  (suma de medianas < suma de medias). **Tweedie** lo reduce (−2% a −10%). La subestimación persistente de
  2026 en exportaciones es una señal de **drift** (§11).
- **Interpretabilidad (SHAP):** la feature dominante es `rel_log_media_12` (**reversión a la media**: series
  sobre su nivel anual tienden a bajar), seguida de la **estacionalidad relativa**, el capítulo y el país.
- **El baseline más "sofisticado" (estacional con tendencia) fue el peor:** extrapolar el crecimiento
  amplifica el ruido.

### 10.2 Problema 2 · Test ene–ago 2026

| | Expo AP | Expo AUC | Expo Brier | Impo AP | Impo AUC | Impo Brier |
|---|---|---|---|---|---|---|
| **LightGBM + grafo** | **0,560** | **0,809** | **0,122** | **0,554** | **0,801** | **0,136** |
| LightGBM sin grafo | 0,558 | 0,804 | 0,123 | 0,552 | 0,797 | 0,137 |
| Frecuencia 12m (mejor baseline) | 0,470 | 0,781 | 0,129 | 0,500 | 0,783 | 0,141 |
| Persistencia | 0,277 | 0,628 | 0,257 | 0,297 | 0,616 | 0,283 |
| Tasa base | 0,200 | | | 0,227 | | |

- El modelo supera al mejor baseline: **AP +19% (expo) y +11% (impo)**, IC de la mejora [+0,086; +0,094] y
  [+0,052; +0,055]. Está **bien calibrado** sin necesidad de recalibrar.
- **El grafo aporta según la historia disponible** (mejora relativa de AP en test):

  | Meses activos en el último año | Exportaciones | Importaciones |
  |---|---|---|
  | 1 | **+8,2%** | **+11,9%** |
  | 2–3 | +1,9% | +3,8% |
  | 4–9 | +0,1% | +0,3% |

  Es coherente con el experimento de *cold start* del §7.4: **el grafo sirve para predecir con poca
  información**. Cuando la serie tiene historia propia, ésta domina.
- **Umbral de decisión:** se deriva de los costos de negocio. El umbral óptimo es costo_FP / (costo_FP +
  costo_FN), y requiere probabilidades calibradas. Con umbral 0,3: precisión ~50% y recall ~60–65%.

### 10.3 Esquemas de división entrenamiento / validación / test (`notebooks/05_esquemas_validacion.ipynb`)

**Concepto:** entrenamiento ajusta los parámetros; validación sirve para tomar decisiones (n.º de árboles,
variantes); test entrega la estimación final y se mira **una vez**. Las proporciones (50/20/30, 40/20/40) son
un trade-off entre la calidad del modelo (más entrenamiento) y la estabilidad de la estimación (más test).
**En series de tiempo la división debe ser cronológica:** un reparto aleatorio pone en entrenamiento filas
cuyo `lag_0` es el objetivo de filas de test, y entrena con meses posteriores a los que predice.

Experimento (P1, L1, h = 1; universo de 20 meses objetivo, ene-2025 a ago-2026; validación usada para
*early stopping*):

| Esquema | Árboles elegidos (expo / impo) | WAPE test expo | WAPE test impo | Meses de test |
|---|---|---|---|---|
| Aleatorio 50/20/30 | 906 / 1.651 | 0,324 | 0,402 | mezclados |
| Aleatorio 40/20/40 | 354 / 1.203 | 0,324 | 0,398 | mezclados |
| Cronológico 50/20/30 | 326 / 628 | 0,308 (walk-forward 0,304) | 0,401 (0,396) | mar–ago 2026 |
| Cronológico 40/20/40 | 202 / 424 | 0,331 (walk-forward 0,319) | 0,416 (0,409) | ene–ago 2026 |

- **La validación aleatoria eligió 1,8–2,8× más árboles:** al estar intercalada con el entrenamiento, no
  detecta el sobreajuste. Las decisiones basadas en ella no son confiables.
- **Resultado honesto:** en las mismas filas de 2026, el reparto aleatorio sobreestimó el desempeño solo
  ~±1 punto porcentual. El modelo no puede explotar el leakage porque no identifica series individuales.
  La garantía del esquema cronológico, en cambio, no depende del diseño de features, y es el único que simula
  el despliegue.
- **La ventana de test pesa más que el porcentaje:** 50/20/30 parece mejor porque su test excluye el shock de
  ene–feb 2026. Con 40/20/40 el modelo fue peor (8 meses de entrenamiento); la mayor estabilidad del test
  solo apareció en importaciones.
- El protocolo principal del proyecto equivale a ~56/19/25 cronológico sobre 32 meses, con reentrenamiento
  mensual (walk-forward).

## 11. Drift monitoring

*Fase CRISP-DM: despliegue (plan de monitoreo y mantenimiento), en un despliegue simulado. Responde la pregunta P4.*

Notebook: `notebooks/06_drift_monitoring.ipynb` · Código: `src/comercio_chile/drift.py` · Figuras `21`–`23`.

**Escenario:** modelo P1 (L1, h = 1) entrenado una vez hasta dic-2025 y desplegado sin reentrenar en
ene–ago 2026 (shock de precios conocido). **Período de control** para medir falsas alarmas: modelo entrenado
hasta jun-2025 y monitoreado en jul–dic 2025.

### 11.1 Cuatro capas, con umbrales definidos antes de mirar el evento

| Capa | Métricas | Umbral |
|---|---|---|
| 0. Calidad de datos | Filas descartadas, valores no convertibles, Δ volumen vs mismo mes del año anterior, % valor con códigos nuevos | Cualquier descarte o conversión fallida, o > 1% de valor nuevo → alerta (detener el pipeline); \|Δ volumen\| > 25% / 50% |
| 1. Data drift | PSI y KS (estadístico D) por feature; mix por capítulo ponderado por valor; índice de precios implícito (valor unitario vs mediana de referencia, ponderado por valor, por segmento) | PSI 0,10 / 0,25; precios ±10% / ±20% (solo commodities) |
| 2. Target drift | PSI del objetivo | 0,10 / 0,25 |
| 3. Desempeño / concept drift | WAPE y sesgo del modelo congelado en un **gráfico de control** (μ, σ del propio modelo en la referencia); brecha congelado vs reentrenado | \|z\| > 2 advertencia; > 3, o 2 seguidas del mismo signo, alerta |

### 11.2 El período de control reveló cinco errores de diseño (corregidos antes de evaluar 2026)
1. **Referencia con NaN estructurales:** `crec_interanual_3m` necesita 15 meses de historia; con orígenes de
   "calentamiento" en la referencia, el PSI daba 4,6 (con la referencia corregida: 0,006).
2. **Features de contexto** (crecimiento del capítulo: ~97 valores por mes, una tasa que cambia por diseño):
   PSI alto incluso sin evento → se monitorea su nivel, no su PSI.
3. **Umbral de sesgo sin calibrar:** un modelo de mediana tiene sesgo normal negativo (−7% a −14%); un umbral
   fijo de ±10% generaba falsas alarmas → gráfico de control calibrado con el propio modelo.
4. **Volumen sin estacionalidad:** enero (temporada de fruta) disparaba una advertencia → se compara con el
   mismo mes del año anterior.
5. **Gráfico de control demasiado sensible para precios:** la referencia se construye relativa a su propia
   mediana y subestima la variación natural (z ≈ 7 sin evento) → tolerancia de negocio de ±10% / ±20%.

### 11.3 Resultados

| Señal | Expo control | Expo 2026 | Impo control | Impo 2026 |
|---|---|---|---|---|
| PSI features por serie (máx.) | 0,08 | 0,07 | 0,05 | 0,04 |
| PSI objetivo (máx.) | 0,01 | 0,01 | 0,00 | 0,02 |
| Mix por valor (máx.) | 0,09 | **0,19** (ene–feb) | 0,05 | 0,07 |
| Índice de precios de commodities | ≤ 8% | **advertencia feb, alerta mar–ago (+34%)** | ≤ 9% | **advertencia abr, alerta may–ago (+37%)** |
| Reglas ingenuas: meses con alerta | **1 (falsa)** | 4 | **2 (falsas)** | 2 |
| Gráfico de control: meses con alerta | 0 | **1 (ene-2026)** | 0 | 0 |
| WAPE congelado vs reentrenado (prom.) | 0,302 vs 0,312 | 0,313 vs 0,318 | 0,412 vs 0,421 | 0,415 vs 0,411 |

**Diagnóstico:**
- **Data drift persistente con concept drift transitorio.** Los precios de commodities subieron hasta +34–37%,
  pero el desempeño del modelo congelado solo se degradó en **enero 2026** (WAPE 0,41, z ≈ 3,7) y se recuperó
  **sin reentrenar**. El modelo congelado y el reentrenado rinden casi igual: las features **relativas al nivel
  reciente** absorben el cambio de nivel de precios en 1–3 meses.
- **El PSI por serie no ve el shock, y eso es correcto:** el shock afecta a pocas series de alto valor, que en
  una métrica por conteo quedan diluidas entre ~2.900 series. Solo las señales **ponderadas por valor** (mix e
  índice de precios) lo detectan.
- **El sesgo es una señal poco sensible** (σ mensual ≈ 8–9 pp en la referencia); la señal útil de desempeño fue
  el WAPE.
- **No todo drift exige reentrenar.** Runbook: alerta de calidad → detener; drift de entradas (precios, mix) →
  informar a negocio y revisar por segmento; alerta de desempeño → reentrenar y comparar; drift de precios +
  desempeño persistente → reentrenar e incorporar precios exógenos.

## 12. Evaluación frente a los objetivos y conclusiones

*Fase CRISP-DM: evaluación (resultados frente a los objetivos de negocio y revisión del proceso).*

| Pregunta | ¿Se respondió? | Evidencia | Límite |
|---|---|---|---|
| P1. Evolución y concentración | Sí | +26% en 2026 explicado por precios (cobre ~+50% en valor unitario, volumen a la baja); 27 productos = 80% del valor exportado | La descomposición precio/volumen solo es fiable en productos homogéneos |
| P2. Capacidades y diversificación | Sí | Comunidades estables que cruzan la clasificación arancelaria; la densidad de relatedness mejora 17% la predicción de nuevos mercados fuera de tiempo | Solo datos de Chile; sin comparación con el resto del mundo |
| P3. Próximos meses y flujos esporádicos | Sí, con un límite claro | WAPE 11–14% menor que el mejor baseline (1–3 meses, ambos flujos, 8 de 8 meses); AP +11–19% en flujos intermitentes | En commodities el modelo apenas mejora: su precio es exógeno a los datos |
| P4. Confiabilidad en producción | Sí, en un despliegue simulado | 0 falsas alarmas en el control; detección del alza de precios y de la única degradación real (ene-2026) | Umbrales estimados con 4–6 meses; sin operación real |
| P5. Efecto de los aranceles de EE.UU. (segunda iteración) | Sí, con un resultado nulo para el 10% | 10%: −12,6% no significativo (IC 95% −32% a +11%; p de permutación 0,52), efecto mínimo detectable ~29%. 50% al cobre semielaborado: −92% en toneladas a EE.UU., sin desvío a otros mercados | Un solo país tratado; ventana de cuatro meses antes de que cambiaran las reglas |
| P6. Qué apoyo escalar (tercera iteración) | Sí, como diseño validado; el efecto es simulado | Tests A/A con datos reales: 6% de falsos positivos con errores agrupados (9% ingenuo) y 17% con revisiones mensuales sin corregir; en 500 repeticiones, estimaciones insesgadas y cobertura ~95% | El tamaño del efecto es un supuesto; no hay un experimento ejecutado |
| Análisis por empresa | No (descartado) | IDs renumerados cada mes (§2.5) | Restricción de la anonimización, no del método |

**Criterios de éxito (§1.3):** los cuatro se cumplieron. Los totales coinciden con el Banco Central, el grafo
supera al baseline sin grafo (IC 95% de la mejora excluye el cero), los modelos superan al mejor baseline en el
test (también con IC 95% que excluye el cero) y el monitoreo cumple la prueba de control y de detección. El
criterio de la segunda iteración (P5) también se cumplió: el diseño principal no rechaza tendencias paralelas, se
sometió a una prueba placebo y el resultado nulo se reporta con su efecto mínimo detectable. El de la tercera (P6)
también: el análisis preregistrado da ~5–6% de falsos positivos en tests A/A y recupera el efecto inyectado.

**Conclusiones:**
1. **Datos:** se construyó un pipeline reproducible para 15,6 M de registros (93 archivos descargados de
   datos.gob.cl), con 14 tipos de defectos reparados, 0 registros perdidos y totales validados contra el Banco
   Central. Se descubrió que los IDs de empresa se renumeran cada mes, lo que redefinió el alcance de todo el
   análisis.
2. **EDA:** el crecimiento exportador de 2026 (+26%) es de **precios de commodities**, no de volumen; las
   exportaciones están extremadamente concentradas y tienen dos regímenes temporales (estacional vs persistente).
3. **Grafo:** una red de capacidades (productos que exportan las mismas empresas) revela ecosistemas que cruzan
   la clasificación arancelaria y **mejora fuera de tiempo** la predicción de nuevos mercados (+17% AP). Su
   aporte se concentra donde falta historia (*cold start*).
4. **Forecasting:** un LightGBM global supera a todos los baselines (WAPE −11% a −14%). Lo decisivo fue la
   **formulación** (pérdida L1 y objetivo sin escala), no los hiperparámetros. Hay un trade-off mediana/media
   entre el error por serie y el total.
5. **Clasificación:** los flujos intermitentes se predicen con AP +11–19% sobre el mejor baseline, con buena
   calibración.
6. **Validación:** las divisiones deben ser cronológicas; la ventana de test importa tanto como su tamaño.
7. **Drift:** un monitoreo de 4 capas, calibrado en un período de control, detectó el shock de 2026 y la única
   degradación real (ene-2026) sin falsas alarmas, y permitió distinguir data drift de concept drift.
8. **Inferencia causal (§15):** el arancel recíproco del 10% no tuvo un efecto distinguible de cero en sus
   primeros cuatro meses (−12,6%, efecto mínimo detectable ~29%), mientras que el 50% al cobre semielaborado
   eliminó el flujo a EE.UU. (−92% en toneladas).
9. **Experimentación (§16):** un experimento A/B/n semi-sintético mostró que aleatorizar por producto, analizar
   con errores agrupados y mirar el resultado una sola vez son condiciones necesarias: sin ellas, los falsos
   positivos suben a 9% y 17%.

**Revisión del proceso:** las decisiones con más impacto no fueron de algoritmo sino de comprensión de los
datos y formulación (granularidad ítem/encabezado, IDs renumerados, escala del objetivo, función de pérdida).
La práctica que más errores evitó fue contrastar cada conclusión con una prueba independiente: cifras
oficiales, períodos de control, tests automatizados y baselines.

## 13. Despliegue y entrega de resultados

*Fase CRISP-DM: despliegue.*

**No hay un despliegue productivo.** No existe un proceso programado que descargue los datos cada mes, ni un
modelo servido por una API, ni un dashboard en operación. El despliegue se **simuló** en `notebooks/06`: un
modelo congelado a diciembre de 2025 opera durante 2026 bajo el sistema de monitoreo, con un runbook de acciones
por tipo de alerta (§11).

**Cómo se entregan los resultados:**
- **Repositorio público** con el pipeline como comandos reproducibles (`download` → `ingest` → `references` →
  `clean`), tests anti-leakage y de validación del estimador causal y del diseño experimental, y los ocho notebooks ejecutados con sus
  resultados visibles.
- **Este informe técnico** y el `README.md` como resumen para un lector nuevo.
- **29 figuras** en `reports/figures/`, pensadas para comunicar los hallazgos fuera del código.
- **Módulo de monitoreo** (`drift.py`) listo para ejecutarse mensualmente sobre datos nuevos.

**Qué faltaría para un despliegue real:** orquestar la ejecución mensual cuando Aduana publica cada archivo
(~1 mes de rezago), registrar versiones de los modelos, automatizar las alertas del runbook y exponer los
pronósticos en un tablero o API para los usuarios de negocio.

## 14. Limitaciones, mejoras y trabajo futuro

*Fase CRISP-DM: evaluación (próximos pasos), que reinicia el ciclo.*

**Limitaciones:**
- Solo 32 meses: los gráficos de control y la estacionalidad se estiman con pocos ciclos.
- Sin precios internacionales: el error en commodities queda dominado por un factor exógeno.
- IDs de empresa sin continuidad: no hay modelamiento por empresa (por diseño de la anonimización).
- Etiquetas de producto derivadas de texto libre; FOB (expo) vs CIF (impo) no son directamente comparables.
- El grafo usa solo exportaciones chilenas (no hay datos mundiales para un Product Space clásico).

**Mejoras y trabajo futuro:**
- Incorporar precios de mercado (cobre LME, litio, celulosa, combustibles) como features y como señal
  anticipada de drift. Es la mejora de mayor impacto esperado.
- *Early stopping* interno por origen (el mínimo de validación estaba en ~180 árboles, no en 500).
- Pronóstico probabilístico (cuantiles) en lugar de puntual; intervalos para los totales.
- Reconciliación jerárquica (serie → capítulo → total) para pronósticos coherentes entre niveles.
- Orquestación del pipeline (descarga mensual automática + monitoreo + reentrenamiento condicionado por alertas).
- Extender el grafo de capacidades con datos de comercio mundial (UN Comtrade) para medir ventaja comparativa.

---

## 15. Inferencia causal: efecto de los aranceles de EE.UU. de 2025

*Segunda iteración del ciclo CRISP-DM, con una pregunta nueva (P5). Notebook: `notebooks/07_inferencia_causal`;
módulo: `causal.py`.*

### 15.1 Por qué una segunda iteración
Los modelos de §8–§10 predicen, y para predecir basta con que una variable anticipe el resultado. Una pregunta
de impacto ("¿cuánto afectó el arancel?") exige otra cosa: comparar lo que pasó con lo que habría pasado sin el
arancel, un contrafactual que nunca se observa. Como no se puede experimentar, se usa un **experimento natural**:
EE.UU. impuso aranceles en fechas conocidas, a un destino y no a los otros, y a unos productos y no a otros.

### 15.2 Comprensión del negocio: las medidas
Para Chile, lo relevante en 2025–2026 fue: el arancel recíproco del 10% (desde el 5-abr-2025, con exenciones
listadas en el Annex II de la EO 14257); el 50% de la Sección 232 al cobre semielaborado (desde el 1-ago-2025,
anunciado el 8-jul; el cátodo quedó exento); tasas mayores para varios competidores desde el 7-ago-2025 (Chile se
mantuvo en 10%); cambios al Annex II (8-sep), la Sección 232 a la madera (14-oct) y la exención agrícola (13-nov);
y el fallo de la Corte Suprema que anuló los aranceles IEEPA (20-feb-2026), reemplazados por un 10% transitorio de
la Sección 122 hasta el 24-jul-2026. Fuentes en el Anexo B.

Consecuencia para el diseño: la **ventana principal es abril–julio de 2025**, cuando casi todos los proveedores
de EE.UU. pagaban el mismo 10%. Después, el tratamiento deja de ser uniforme y los meses solo se describen.

### 15.3 Preparación de los datos
- **Panel balanceado** producto (HS6) × destino × mes con **ceros explícitos** (848 mil filas; 73% ceros). Si el
  arancel hace desaparecer un flujo, ese cero es el efecto. Entran los pares con comercio *antes* del arancel:
  elegir la muestra con meses posteriores la condicionaría al resultado (verificado en `tests/test_causal.py`).
- **Clasificación de productos** con el Annex II oficial (PDF de la Casa Blanca → 1.039 subpartidas de 8 dígitos
  de EE.UU. → 610 HS6). Si el anexo lista la subpartida completa (`hs6 + "00"`), el HS6 es exento; si lista solo
  algunas aperturas, es **exento parcial** y se excluye, porque no se sabe qué parte del flujo pagó.
- **Exclusiones:** acero, aluminio y automóviles (Sección 232 propia), cobre (Sección 232 desde agosto y arbitraje
  del cátodo durante la investigación) y metales preciosos (dudas arancelarias sobre el oro). El cátodo es el
  principal producto chileno en EE.UU., pero mezclar su arbitraje con el arancel recíproco contaminaría la
  estimación.
- **Seis celdas con valor negativo** (correcciones de declaraciones, US$ 1.300 en total) se llevan a cero: PPML
  exige valores no negativos.

### 15.4 Modelado: de la triple a la doble diferencia
**Estimador: PPML** (Poisson pseudo-máxima verosimilitud con efectos fijos, `pyfixest`), el estándar para flujos
de comercio: admite ceros, es consistente bajo heterocedasticidad (Santos Silva y Tenreyro, 2006) y pondera por
valor. Errores estándar agrupados por producto.

**Diseño planeado: triple diferencia** (EE.UU. vs otros destinos × gravados vs exentos × antes vs después), con
efectos fijos par, producto × mes y destino × mes. En teoría separa el arancel de cualquier shock propio de EE.UU.
**No pasó su contraste:** el test conjunto de pretendencias rechaza (p = 0,001). Chile vende a EE.UU. solo 40
productos exentos, dominados por embarques irregulares de madera, yodo y litio, y la brecha gravados–exentos ya
se movía antes del arancel.

**Diseño principal: doble diferencia** con los productos gravados (EE.UU. vs otros destinos del mismo producto),
con efectos fijos producto × destino (nivel de cada flujo) y producto × mes (precios, cosechas, oferta). Sus
pretendencias no rechazan al 5% (p = 0,08 con todo el período; aval débil, porque el test tiene poca potencia).
Ambos diseños dan casi la misma estimación puntual, así que el cambio no responde a buscar un resultado.

**Alternativas descartadas:**
- *Antes/después simple:* confunde el arancel con todo lo demás que cambió (precios, demanda mundial).
- *Regresión en log(1 + y):* el resultado depende de la unidad de medida y está sesgado con heterocedasticidad.
- *Control sintético para el arancel recíproco:* hay cientos de unidades tratadas (productos), no una; el
  panel con efectos fijos aprovecha mejor esa estructura.

### 15.5 Evaluación

| Especificación | Efecto | IC 95% | p |
|---|---|---|---|
| **Principal: doble diferencia, abr–jul 2025** | **−12,6%** | [−31,5%, +11,4%] | 0,28 |
| Sin ene–mar 2025 (posible anticipación) | −12,1% | [−30,1%, +10,5%] | 0,27 |
| Ventana abr–ago 2025 | −12,6% | [−30,6%, +10,0%] | 0,25 |
| Sin filete de salmón (mayor flujo) | −13,6% | [−32,9%, +11,4%] | 0,26 |
| Sin salmón (partida 0304) | −21,2% | [−38,2%, +0,5%] | 0,06 |
| Triple diferencia (pretendencias rechazadas) | −12,7% | [−39,3%, +25,4%] | 0,46 |

- **Event study:** febrero de 2025 muestra un salto de +28% (adelanto de embarques, el único mes significativo);
  abril–julio, coeficientes entre −5% y −10%, ninguno significativo.
- **Placebo en otros destinos:** asignando un arancel ficticio a cada uno de los 30 mayores destinos, la mitad
  muestra un cambio igual o más extremo que EE.UU. (p de permutación = 0,52). Con un solo país tratado, esta
  prueba es más confiable que los errores agrupados.
- **Potencia:** el efecto mínimo detectable (80% de potencia, 5% de significancia) es ~−29%. El diseño descarta
  una caída grande e inmediata, no una moderada.
- **Validación del estimador:** en datos simulados con la estructura del panel, la doble diferencia recupera
  efectos conocidos de −30%, −10% y 0% (`tests/test_causal.py`).
- **Sin salmón** la caída es mayor (p = 0,06), pero no se trata como hallazgo: después de varias
  especificaciones, un p de 0,06 en una de ellas es esperable por azar.

**Lectura económica:** un efecto pequeño es plausible. Entre abril y julio casi todos los proveedores pagaban el
mismo 10%, así que el comprador estadounidense no tenía a quién cambiarse; la ventana es corta para reorganizar
contratos; y el valor FOB no cambia si el importador absorbe el arancel.

### 15.6 Estudio de caso: 50% al cobre semielaborado
Afecta casi solo al alambre de cobre (7408), así que no hay panel para una diferencia en diferencias. Se mide en
**toneladas** (en dólares, el alza del cobre de 2026 se confundiría con volumen), comparando ene-2024–jul-2025 con
ago-2025–ago-2026, con intervalos por bootstrap de bloques de 3 meses (conserva la autocorrelación mensual).

| Destino | Antes (t/mes) | Después (t/mes) | Cambio | IC 95% |
|---|---|---|---|---|
| EE.UU. | 720 | 58 | −92% | [−819, −503] t/mes |
| Resto del mundo | 2.898 | 2.701 | −7% | [−763, +615] t/mes |
| Total | 3.617 | 2.759 | −24% | [−1.471, +25] t/mes |

El flujo a EE.UU. prácticamente desapareció, después de un máximo en julio de 2025 (adelanto de embarques entre
el anuncio y la vigencia). No hay evidencia de que el volumen se recolocara en otros mercados. La fuerza de esta
evidencia es la nitidez del corte, no la sofisticación del modelo; un control sintético agregaría poco frente a
una caída de 92% en la fecha exacta de vigencia.

### 15.7 Limitaciones
- Un solo país tratado: la inferencia del arancel recíproco descansa en una comparación con un único EE.UU.
- Tendencias paralelas con un aval débil; la triple diferencia, que habría controlado shocks propios de EE.UU.,
  no pasó su contraste.
- Desvío de comercio: si Chile vendió más a otros destinos por el arancel, el grupo de control también cambió
  (violación de SUTVA) y la caída en EE.UU. queda sobrestimada.
- Valor FOB: no muestra quién pagó el arancel ni el precio final en EE.UU.
- Clasificación aproximada: el Annex II está en 8 dígitos de EE.UU. y los datos en 6 dígitos del Sistema
  Armonizado; los casos ambiguos se excluyeron.

### 15.8 Conclusiones
1. **10% recíproco:** caída estimada de ~12% en abril–julio de 2025, no distinguible de cero; se descarta un efecto
   mayor a ~29%.
2. **50% al cobre semielaborado:** efecto inequívoco, −92% en toneladas a EE.UU., sin desvío a otros mercados.
3. **La dosis importa:** un 10% que pagan casi todos los competidores casi no se nota; un 50% específico elimina el
   flujo.
4. **Lección de método:** el diseño más completo en el papel no siempre es el mejor. Lo que decide es si su
   supuesto se sostiene en los datos, y eso se revisa antes de interpretar el resultado.

---

## 16. Experimento A/B/n semi-sintético: qué apoyo convierte una oportunidad en exportación

*Tercera iteración del ciclo CRISP-DM (P6). Notebook: `notebooks/08_experimento_ab`; módulo: `experiment.py`.*

### 16.1 Por qué semi-sintético
Los datos de Aduana son observacionales: no hay ninguna intervención asignada al azar, así que no permiten un
A/B test real. Lo que sí permiten es construir uno **semi-sintético** donde todo lo que depende del ruido de los
datos es real y solo el efecto del tratamiento es un supuesto:

| Componente | Origen |
|---|---|
| 10.000 oportunidades (pares producto–país no exportados en 2025, mejor rankeados por el modelo con grafo de §7.4) | Real |
| Covariables previas (puntaje, comercio del par en 2024 y 2025) | Real |
| Resultado sin tratamiento Y(0): exportar ≥ US$ 10.000 en ene–ago 2026, con su trayectoria mensual | Real |
| Asignación a brazos | Aleatoria |
| Efecto de cada tratamiento | Supuesto, inyectado sobre Y(0): B 0%, C +30%, D +40% relativo |

Como el efecto verdadero se conoce, el procedimiento se puede validar (sesgo, cobertura, potencia, error tipo I).
El notebook no afirma que ninguna campaña funcione.

### 16.2 Diseño (preregistro)
- **Brazos:** A control; B alerta informativa (US$ 50); C rueda de negocios (US$ 1.500); D rueda + misión
  comercial (US$ 4.000). Los costos son supuestos.
- **Métrica principal:** activación binaria. El valor exportado se descartó por su cola pesada (mediana ~US$ 33 mil
  vs promedio ~US$ 195 mil por par activado).
- **Unidad de aleatorización: el producto (HS6)**, no el par. Las campañas de un producto llegan a los mismos
  exportadores en todos los destinos, así que aleatorizar pares contaminaría el control. Aleatorización
  estratificada por puntaje.
- **Costo de aleatorizar por grupos:** ICC por producto = 0,026 y tamaño efectivo de grupo 14,4 → efecto de
  diseño 1,34 (un tercio más de muestra).
- **CUPED** con covariables previas: R² = 6,8%. Con una métrica binaria y poco frecuente, el aporte es modesto.
- **Asignación** √3 : 1 : 1 : 1 (control más grande, porque participa en las tres comparaciones).
- **Comparaciones múltiples:** Holm (familia de 3 comparaciones contra el control); Dunnett como contraste.
- **Potencia:** con tres tratamientos, el efecto mínimo detectable planificado es ~31% relativo, cercano al 30% que
  interesa al negocio. Un cuarto tratamiento ya no habría alcanzado; distinguir C de D tampoco.
- **Detención:** el resultado se mira una sola vez, al final.

### 16.3 Validación del procedimiento con datos reales
| Prueba | Resultado |
|---|---|
| 1.000 tests A/A, análisis ingenuo (pares independientes) | 8,9% de falsos positivos (debería ser 5%) |
| 1.000 tests A/A, errores agrupados por producto | 6,1% |
| SRM sobre los pares en 300 aleatorizaciones correctas | Falsa alarma (p < 0,001) en el 38% |
| SRM sobre los productos (unidad aleatorizada) | 0% de falsas alarmas |
| Revisar el resultado cada mes con 1,96 (*peeking*), 1.000 A/A con trayectorias reales | 17,0% de falsos positivos |
| Revisar cada mes con fronteras de O'Brien–Fleming | 5,6% |

### 16.4 Resultados (simulados)
Test global: p < 0,001. Con CUPED y Holm: C +3,8 pp (+38% relativo) y D +4,4 pp (+43%), ambos con p ajustado
< 0,001; B +0,5 pp, no significativo. Sin ajuste, C aparecía en +48%: el brazo había recibido por azar
oportunidades con algo más de comercio previo (diferencia estandarizada 0,10), y CUPED corrigió ese desbalance.

**Monte Carlo (500 repeticiones del experimento completo):** estimaciones insesgadas en los tres brazos,
cobertura de los IC 95% entre 94% y 96%, potencia de 93% para C con CUPED (87% sin ajuste) y error tipo I de 5,4%
en el brazo sin efecto.

**Decisión:** costo por activación adicional ~US$ 39 mil para C (IC US$ 27–72 mil) y ~US$ 92 mil para D. Como el
diseño no puede distinguir C de D, se escala la opción más barata (C); B no tiene evidencia de efecto.

### 16.5 Limitaciones
- El efecto y su forma (proporcional a la probabilidad base) son supuestos; el notebook valida el método, no las
  campañas.
- Aleatorizar por producto no evita el contagio entre productos relacionados (que comparten exportadores según
  la red de §7); aleatorizar por comunidad de la red lo evitaría con mucha más varianza.
- Sin métricas de resguardo reales (satisfacción, costo operativo), que no existen en estos datos.

---

## Anexo A. Reproducibilidad

Requisitos: Python 3.11 o superior y UnRAR (incluido en WinRAR en Windows; paquete `unrar` en Linux y macOS, o la
variable de entorno `UNRAR` con la ruta al ejecutable).

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

python -m comercio_chile.download    # ~1 GB desde datos.gob.cl
python -m comercio_chile.ingest      # archivos crudos -> data/interim (Parquet)
python -m comercio_chile.references  # tablas de códigos de Aduana
python -m comercio_chile.clean       # panel producto × país × mes -> data/processed
python -m comercio_chile.causal      # lista de exenciones del Annex II (EE.UU.) por HS6
python -m pytest tests

python -m ipykernel install --user --name comercio-chile
jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb
```

La ingesta completa toma unos 30 minutos; el notebook de regresión, unos 25, y el de inferencia causal y el del experimento, unos 5 cada uno.

## Anexo B. Fuentes

- Diccionario de datos DUS/DIN: https://datos.gob.cl/dataset/diccionario-de-datos-para-datos-abiertos-aduana
- Registros de exportación/importación: https://datos.gob.cl/organization/servicio_nacional_de_aduanas
- Tablas de códigos (Anexo 51): https://www.aduana.cl/compendio-de-normas-anexo-51/aduana/2009-11-19/163937.html
- EO 14257 (arancel recíproco) y su Annex II: https://www.whitehouse.gov/wp-content/uploads/2025/04/eo-14257.pdf,
  https://www.whitehouse.gov/wp-content/uploads/2025/04/Annex-II.pdf
- Sección 232 al cobre: https://www.whitecase.com/insight-alert/president-trump-orders-50-percent-section-232-tariff-copper-imports
- EO 14326 (tasas desde el 7-ago-2025): https://www.whitehouse.gov/presidential-actions/2025/07/further-modifying-the-reciprocal-tariff-rates/
- EO 14346 (cambios al Annex II): https://www.presidency.ucsb.edu/documents/executive-order-14346-modifying-the-scope-reciprocal-tariffs-and-establishing-procedures
- Sección 232 a la madera: https://www.internationaltradeinsights.com/2025/09/section-232-tariffs-on-timber-lumber-and-certain-wood-products-take-effect-on-october-14/
- Exención agrícola: https://www.federalregister.gov/documents/2025/11/25/2025-21203/modifying-the-scope-of-the-reciprocal-tariffs-with-respect-to-certain-agricultural-products
- Fallo de la Corte Suprema sobre IEEPA: https://www.hklaw.com/en/insights/publications/2026/02/supreme-court-strikes-down-ieepa-tariffs
- Sección 122: https://globaltradealert.org/blog/from-ieepa-to-section-122
- Santos Silva, J. y Tenreyro, S. (2006). The Log of Gravity. *The Review of Economics and Statistics*, 88(4).
