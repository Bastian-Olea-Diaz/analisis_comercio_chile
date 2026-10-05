# Comercio exterior de Chile: análisis, grafos, forecasting y monitoreo de drift

Informe técnico del proyecto. Registra qué se hizo, por qué y qué alternativas se descartaron en cada etapa.
Los resultados numéricos provienen de los notebooks en `notebooks/`.

---

## 1. Contexto y objetivo

Chile es una economía muy abierta al comercio: cobre, litio, fruta, salmón y celulosa explican gran parte de
sus exportaciones, y depende de importaciones de combustibles, maquinaria y bienes de consumo. El Servicio
Nacional de Aduanas publica cada declaración de exportación (DUS) e importación (DIN) a nivel de ítem como
datos abiertos.

**Objetivo del proyecto:** construir un flujo completo de Data Science sobre esos registros:

1. Entender la estructura y la calidad de los datos (EDA).
2. Representar el comercio como un grafo cuando aporte información que una tabla no entrega.
3. Predecir el comportamiento de los meses siguientes (regresión y/o clasificación, según lo que permitan los datos).
4. Evaluar los modelos contra baselines con validación temporal honesta.
5. Monitorear drift en datos y en concepto, con umbrales y criterios de alerta definidos.

---

## 2. Datos

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

## 12. Conclusiones

1. **Datos:** se construyó un pipeline reproducible para 15,6 M de registros (77 archivos descargados + 32
   entregados), con 14 tipos de defectos reparados, 0 registros perdidos y totales validados contra el Banco
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

## 13. Limitaciones, mejoras y trabajo futuro

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

## Anexo A. Reproducibilidad

Requisitos: Python 3.11+ y UnRAR (incluido en WinRAR; en Linux/macOS, paquete `unrar`). Ver el README para
los comandos completos. Orden del pipeline: `download` → `ingest` → `references` → `clean` → tests →
notebooks 01–06.

## Anexo B. Fuentes

- Diccionario de datos DUS/DIN: https://datos.gob.cl/dataset/diccionario-de-datos-para-datos-abiertos-aduana
- Registros de exportación/importación: https://datos.gob.cl/organization/servicio_nacional_de_aduanas
- Tablas de códigos (Anexo 51): https://www.aduana.cl/compendio-de-normas-anexo-51/aduana/2009-11-19/163937.html
