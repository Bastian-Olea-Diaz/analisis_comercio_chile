# Comercio exterior de Chile: análisis, grafos, forecasting y monitoreo de drift

Proyecto de Data Science de punta a punta sobre los registros de exportaciones e importaciones del Servicio
Nacional de Aduanas de Chile: 15,6 millones de ítems de declaraciones aduaneras entre enero de 2024 y agosto
de 2026. Abarca la ingesta y validación de los datos crudos, el análisis exploratorio, una red de capacidades
productivas, modelos de pronóstico y clasificación con validación temporal, y un sistema de monitoreo de drift.

El detalle de cada decisión técnica, con las alternativas descartadas, está en [`docs/PROYECTO.md`](docs/PROYECTO.md).

## Idea del proyecto

Chile es una economía pequeña y muy abierta: el comercio de bienes equivale a más de la mitad de su PIB, y sus
exportaciones dependen de pocos productos (cobre, litio, fruta, salmón, celulosa) y de pocos mercados. Eso hace
que la economía sea sensible a los precios internacionales y a la demanda de socios como China.

El proyecto busca responder, con los registros oficiales de Aduana, cuatro preguntas planteadas desde la
perspectiva de un analista de comercio exterior o una agencia de promoción de exportaciones:

1. Qué explica la evolución reciente del comercio chileno y cuán concentrado está.
2. Qué productos comparten capacidades productivas y hacia qué mercados puede diversificarse Chile.
3. Cuánto se comerciará en los próximos meses, a nivel de producto y país, y si un flujo esporádico se
   repetirá.
4. Cómo detectar cuándo un modelo en producción deja de ser confiable porque los datos cambiaron.

## Metodología: CRISP-DM

El proyecto siguió el ciclo CRISP-DM (*Cross-Industry Standard Process for Data Mining*):

| Fase | En este proyecto | Dónde |
|---|---|---|
| Comprensión del negocio | Cuatro preguntas, sus objetivos analíticos y criterios de éxito fijados antes de evaluar: superar a baselines simples fuera de tiempo, cuadrar los totales con cifras oficiales y monitorear sin falsas alarmas | `docs/PROYECTO.md` §1 |
| Comprensión de los datos | Contraste de la documentación con los datos, verificación de calidad de 15,6 M de registros y análisis exploratorio | `ingest.py`, `01_eda` |
| Preparación de los datos | Reparación de registros, imputación exacta de valores truncados, panel producto × país × mes y features sin leakage | `clean.py`, `features.py` |
| Modelado | Red de capacidades productivas, modelo global de pronóstico y clasificador de flujos intermitentes, comparados contra baselines | `02_grafo`, `03_regresion`, `04_clasificacion` |
| Evaluación | Test fuera de tiempo evaluado una sola vez, intervalos de confianza, tests anti-leakage, comparación de esquemas de validación y contraste de cada resultado con las preguntas iniciales | `05_esquemas_validacion`, `docs/PROYECTO.md` §10 y §12 |
| Despliegue | Despliegue simulado: un modelo congelado opera durante 2026 bajo un sistema de monitoreo con umbrales y un runbook de acciones. Los resultados se entregan como repositorio reproducible, notebooks, figuras e informe técnico | `06_drift_monitoring`, `drift.py` |

El ciclo fue iterativo. Algunos hallazgos obligaron a volver a fases anteriores:

- **Datos → negocio:** con solo 8 meses de 2026 no era posible modelar la estacionalidad, por lo que el alcance se
  amplió a 2024–2026. Al descubrir que los IDs de empresa se renumeran cada mes, se descartaron las preguntas por
  empresa.
- **Modelado → preparación:** un modelo global en escala absoluta subestimaba las series más grandes; el objetivo y
  las features se reformularon como cambios relativos al nivel reciente.
- **Evaluación → modelado:** el primer modelo perdía contra una media móvil; la pérdida se cambió a L1 y se agregó
  un modelo Tweedie para los totales. Un test automatizado detectó una fuga de información, que se corrigió.
- **Despliegue → evaluación:** el período de control del monitoreo reveló cinco errores de diseño, corregidos
  antes de evaluar 2026.

No hay un despliegue productivo (proceso programado, API o dashboard en operación). Esa etapa se cubrió con el
despliegue simulado y con los entregables del repositorio; lo que faltaría para operarlo está descrito en el
informe técnico (§13).

## Datos

| Fuente | Contenido |
|---|---|
| [datos.gob.cl, Servicio Nacional de Aduanas](https://datos.gob.cl/organization/servicio_nacional_de_aduanas) | Declaraciones de exportación (DUS, 84 columnas) e importación (DIN, 178 columnas), un archivo por mes, ene-2024 a ago-2026. 93 archivos comprimidos, ~1 GB |
| [Diccionario de datos DUS/DIN v2.0](https://datos.gob.cl/dataset/diccionario-de-datos-para-datos-abiertos-aduana) | Nombres y tipos de columnas (`references/`) |
| [Compendio de Normas Aduaneras, Anexo 51](https://www.aduana.cl/compendio-de-normas-anexo-51/aduana/2009-11-19/163937.html) | Tablas de códigos: países, aduanas, puertos, tipos de operación |

Cada fila de los archivos es un ítem de una declaración. Los identificadores de empresa vienen anonimizados.

## Métodos

### Ingesta y calidad de datos

Los archivos son texto sin encabezado, con codificación latin-1 y formatos de compresión mixtos (.rar, .zip,
multivolumen). La ingesta lee todo como texto, tipa cada columna y **cuenta** cada valor que no se puede
convertir, de modo que nada se descarta en silencio. Durante el proceso se detectaron y corrigieron defectos
reales en los archivos de origen:

- Registros partidos en dos líneas por un salto de línea dentro de un campo, y retornos de carro sueltos
  dentro de nombres de puertos.
- Registros truncados en origen: se verificó que el corte ocurre al final de la línea (sin desplazamiento de
  columnas), y el valor perdido se reconstruye exactamente a partir del total de la declaración.
- Montos de encabezado repetidos en cada ítem: sumarlos infla los totales 1,8 veces. Se agrega siempre a nivel
  ítem, y el total de exportaciones de 2024 (US$ 99.600 millones) coincide con la cifra del Banco Central.
- La metadata publicada junto al dataset de exportaciones describe en realidad las importaciones (se usó el
  diccionario oficial v2.0), y el "número único de exportador" resultó
  **renumerarse cada mes** (verificado con la comuna de la empresa como atributo de control). Por eso el análisis
  se construye sobre dimensiones estables (producto, país), y no por empresa.

Resultado: 0 registros perdidos y un panel mensual producto (HS6) × país como unidad de análisis.

### Análisis exploratorio

Descomposición del crecimiento en efecto precio y efecto volumen, concentración por producto y por empresa
(participación del top 10 e índice HHI), estacionalidad por sector (autocorrelación con rezagos de 1 y 12 meses)
e intermitencia de las series producto–país.

### Red de capacidades productivas

Dos productos se conectan cuando **las mismas empresas los exportan** en el mismo mes (*relatedness*, en la línea
del Product Space de Hidalgo y Hausmann). Las empresas que exportan decenas de productos se ponderan por
1/(k−1) para que no dominen la red. Las comunidades se detectan con Louvain, controlando su estabilidad entre
semillas, y la utilidad del grafo se evalúa con una prueba predictiva fuera de tiempo: si la densidad de
productos relacionados anticipa qué pares producto–país nuevos empieza a exportar Chile.

### Modelos predictivos

- **Regresión:** valor del mes siguiente (horizontes de 1 a 3 meses) para ~16 mil series producto–país regulares,
  con un **modelo global LightGBM** por flujo. El objetivo es el cambio relativo al nivel reciente, sin escala,
  con pérdida L1 (mediana); un segundo modelo Tweedie estima la media para los totales agregados.
- **Clasificación:** probabilidad de que una serie intermitente tenga comercio el mes siguiente, con features de
  actividad y del grafo de capacidades.
- **Validación walk-forward** con reentrenamiento mensual: en cada mes solo se usa información disponible a esa
  fecha. Validación en jul–dic 2025 y test en ene–ago 2026, evaluado una sola vez.
- **Tests automatizados anti-leakage:** alteran los datos posteriores a cada fecha de corte y verifican que las
  features no cambien. Uno de ellos detectó y permitió corregir una fuga de información real.
- Comparación contra baselines (naive, naive estacional, media móvil), intervalos de confianza por bootstrap,
  curvas de aprendizaje y de complejidad, calibración e interpretación con valores SHAP.
- Comparación de divisiones entrenamiento/validación/test (50/20/30 y 40/20/40) aleatorias frente a cronológicas.

### Monitoreo de drift

Sistema de cuatro capas (calidad de datos, data drift, target drift y desempeño) sobre un modelo congelado a
diciembre de 2025 y desplegado en 2026. Usa PSI, el estadístico KS, métricas ponderadas por valor, un índice
de precios implícito y gráficos de control calibrados con el comportamiento del propio modelo. Los umbrales se
ajustaron en un **período de control** sin eventos, antes de evaluar el período con un shock de precios conocido.

## Resultados

Los resultados siguen el orden de las cuatro preguntas. La evaluación frente a cada una, con sus límites, está
en el informe técnico (§12).

**El crecimiento exportador de 2026 es de precios, no de volumen.** Las exportaciones de bienes crecieron 26% en
enero–agosto de 2026. En los cátodos de cobre, el valor unitario subió cerca de 50% sobre el promedio de 2024,
mientras el volumen cayó. El litio y los metales preciosos duplicaron su valor; la fruta, la celulosa y el vino
cayeron.

![Cobre: precio vs volumen](reports/figures/03_cobre_precio_volumen.png)

**Las exportaciones están mucho más concentradas que las importaciones.** 27 productos explican el 80% del valor
exportado (en importaciones se necesitan 443), y China recibe el 37%.

**La red de capacidades revela ecosistemas productivos que cruzan la clasificación arancelaria**: cobre con oro
(subproducto de la misma minería), litio con nitrato de potasio (mismos salares) y celulosa con cartulina. La
estructura es fuerte y estable (modularidad 0,76; ARI entre semillas 0,90). Validada fuera de tiempo, la red
mejora en 17% la predicción de nuevos mercados de exportación sobre un baseline sin grafo: de los 100 pares
producto–país mejor rankeados que Chile no exportaba en 2025, el 31% se exportó en 2026, frente a ~1% al azar.

![Red de capacidades productivas](reports/figures/09_red_capacidades.png)

**El modelo de pronóstico supera a todos los baselines en el test de 2026.** Su error ponderado por valor (WAPE)
es 11–14% menor que el del mejor baseline en los tres horizontes y en ambos flujos, y es mejor en los 8 meses
evaluados (por ejemplo, exportaciones a 1 mes: 0,319 frente a 0,362). La mejora decisiva vino de la
**formulación** (pérdida alineada con la métrica y objetivo sin escala), no del ajuste de hiperparámetros: llevó
el WAPE de 0,59 a 0,31. En commodities el modelo apenas supera al baseline, porque su precio es exógeno a los
datos.

![Error por horizonte](reports/figures/14_wape_horizonte.png)

**Flujos intermitentes.** El clasificador mejora la *average precision* en 19% (exportaciones) y 11%
(importaciones) sobre el mejor baseline, con probabilidades bien calibradas. El aporte del grafo se concentra en
las series con poca historia propia (+8% y +12% en las que tuvieron un solo mes con comercio).

**Monitoreo de drift.** Tras corregir cinco errores de diseño detectados en el período de control, el sistema no
generó falsas alarmas, detectó el alza de precios de 2026 y la única degradación real del modelo (enero de 2026).
El diagnóstico fue un data drift persistente con un concept drift transitorio: gracias a sus features relativas,
el modelo congelado se recuperó sin reentrenar y rindió igual que uno reentrenado mensualmente.

![Panel de monitoreo](reports/figures/22_panel_drift_exportaciones.png)

## Tecnologías

Python 3.13 · pandas · NumPy · PyArrow (Parquet) · LightGBM · scikit-learn · SciPy · NetworkX · Matplotlib ·
Jupyter · pytest · requests (API CKAN de datos.gob.cl)

## Estructura del proyecto

```
├── src/comercio_chile/        Paquete del proyecto
│   ├── download.py            Descarga desde datos.gob.cl con manifiesto (URL, tamaño, SHA-256)
│   ├── ingest.py              Archivos crudos -> Parquet tipado, con reparación y log de calidad
│   ├── schema.py              Esquemas DUS/DIN leídos del diccionario oficial
│   ├── references.py          Tablas de códigos de Aduana (Anexo 51)
│   ├── clean.py               Reglas de limpieza y panel producto × país × mes
│   ├── graph.py               Red de capacidades, densidad de relatedness, comunidades
│   ├── activation.py          Prueba predictiva del grafo (nuevos pares producto–mercado)
│   ├── features.py            Features temporales sin leakage
│   ├── forecast.py            Baselines, modelo global y validación walk-forward
│   ├── classify.py            Clasificación de flujos intermitentes
│   ├── splits.py              Divisiones entrenamiento/validación/test por porcentaje
│   ├── drift.py               Monitoreo de drift en cuatro capas
│   └── data.py, config.py, viz.py
├── notebooks/                 Análisis y resultados, ejecutados en orden (01 a 06)
├── tests/                     Tests anti-leakage
├── references/                Diccionario oficial y tablas de códigos
├── reports/figures/           Figuras generadas por los notebooks
└── docs/PROYECTO.md           Informe técnico: decisiones, alternativas y resultados
```

| Notebook | Contenido |
|---|---|
| `01_eda` | Calidad de los datos, composición, concentración, estacionalidad, precio vs volumen |
| `02_grafo` | Red de capacidades, comunidades, red bipartita ecosistema–país, validación predictiva |
| `03_regresion` | Pronóstico del valor mensual: experimentación, sobreajuste, test y SHAP |
| `04_clasificacion` | Clasificación de flujos intermitentes, calibración y umbrales de decisión |
| `05_esquemas_validacion` | Divisiones aleatorias vs cronológicas (50/20/30, 40/20/40) |
| `06_drift_monitoring` | Calidad de datos, data drift, target drift y desempeño del modelo congelado |

## Reproducción

Requisitos: Python 3.11 o superior y UnRAR (incluido en WinRAR en Windows; paquete `unrar` en Linux y macOS,
o la variable de entorno `UNRAR` con la ruta al ejecutable).

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

python -m comercio_chile.download    # ~1 GB desde datos.gob.cl
python -m comercio_chile.ingest      # archivos crudos -> data/interim (Parquet)
python -m comercio_chile.references  # tablas de códigos de Aduana
python -m comercio_chile.clean       # panel producto × país × mes -> data/processed
python -m pytest tests

python -m ipykernel install --user --name comercio-chile
jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb
```

La ingesta completa toma unos 30 minutos, y el notebook de regresión, que reentrena el modelo para cada mes y
horizonte, alrededor de 25.

## Licencia de los datos

Los datos son publicados por el Servicio Nacional de Aduanas de Chile en el Portal de Datos Abiertos del
Gobierno de Chile, bajo licencia Creative Commons Atribución.
