# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Modelo predictivo para la estimación del riesgo y la severidad de incendios forestales basado en datos meteorológicos e históricos mediante algoritmos de aprendizaje automático supervisado. 
Procesamiento de datos empresariales/ambientales estructurados (Datos tabulares) y modelado predictivo de clasificación/regresión.

 Objetivos de aprendizaje
1. Fase de ingeniería de datos: Aprender a procesar y limpiar grandes 
volúmenes de datos tabulares, gestionando valores nulos, registros 
duplicados y codificación de variables categóricas.
2. Análisis Exploratorio de Datos (EDA): Desarrollar habilidades avanzadas 
de minería de datos para identificar correlaciones ocultas entre factores 
meteorológicos y la propagación del fuego.
3. Modelado predictivo: Implementar, optimizar y comparar algoritmos de 
aprendizaje supervisado avanzados (como Random Forest, XGBoost y 
LightGBM) orientados tanto a la clasificación del riesgo (Bajo/Medio/Alto) 
como a la regresión (predicción de hectáreas afectadas).
4. Explicabilidad del modelo (XAI): Aplicar técnicas para la interpretación de 
modelos (tales como Feature Importance o valores SHAP) con el fin de 
determinar científicamente qué variables climáticas tienen mayor peso en la 
criticidad del incendio.
5. Evaluación métrica formal: Dominar la evaluación de modelos mediante el 
uso de matrices de confusión, curvas ROC-AUC, precisión, Recall y F1
Score en entornos de datos desbalanceados.
5. Resultados esperados
• Un pipeline completo en Python que realice de forma automatizada la 
ingesta, limpieza y transformación de los datos ambientales.
• Un modelo de aprendizaje automático optimizado capaz de clasificar el nivel 
de riesgo de una zona geográfica con una métrica de rendimiento F1-Score 
superior al 80%.
• Un análisis comparativo documentado que demuestre cuál de los algoritmos 
evaluados ofrece mejor capacidad de generalización frente a datos no vistos.
• Un informe técnico detallado sobre la importancia de las características, 
concluyendo cuáles son las condiciones meteorológicas e históricas que 
disparan exponencialmente el riesgo de incendios de gran magnitud.
6. Asignaturas/módulos relacionados con los objetivos y resultados
• Aprendizaje Automático Supervisado (Machine Learning).
• Preprocesamiento de Datos, Análisis Exploratorio y Minería de Datos.
• Evaluación, Optimización y Métricas de Modelos Predictivos.
• Programación Avanzada en Python para Ciencia de Datos.

## Data sources

 Métodos, materiales y tecnologías de uso potencial
• Lenguaje de programación: Python 3.x.
• Entorno de desarrollo: Jupyter Notebooks 
• Librerías principales:
• Manipulación de datos: Pandas y NumPy.
• Visualización: Matplotlib y Seaborn.
• Modelado clásico y métricas: Scikit-Learn.
• Modelado avanzado (Boosting): XGBoost y LightGBM.
• Origen de los datos y Viabilidad:

Two independent datasets, each with its own EDA notebook and no shared code between them:

- **`DataSets/FW_Veg_Rem_Combined.csv`** — 55,367 wildfire records (1991–2015) combining US Forest Service
  fire incidents, NOAA weather data, and USGS vegetation classification. Source:
  https://www.kaggle.com/datasets/capcloudcoder/us-wildfire-data-plus-other-attributes
  Columns include `fire_size`, `fire_size_class` (A–G), `stat_cause_descr`, `latitude`/`longitude`, `state`,
  discovery/containment dates, and pre-fire weather windows at -30/-15/-7 days and at containment
  (`Temp_*`, `Wind_*`, `Hum_*`, `Prec_*`). **Missing weather values are encoded as `-1`, not NaN or null** —
  any analysis or dashboard code touching these columns must treat `-1` as missing.

Las columnas del fichero CSV han sido renombradas: 
'Unnamed': '0.1', 
'Unnamed': '0',
'fire_name': 'Name of Fire',
'fire_size': 'Size of Fire',
'fire_size_class': 'Class of Fire Size (A-G)', 
'stat_cause_descr': 'Cause of Fire', 
'latitude': 'Latitude of Fire', 
'longitude': 'Longitude of Fire', 
'state': 'State of Fire',
'disc_clean_date': 'Discovery Date', 
'cont_clean_date': 'Containment Date', 
'discovery_month': 'Month of Discovery',
'disc_date_final': 'Final Discovery Date', 
'cont_date_final': 'Final Containment Date', 
'putout_time': 'Time to Put Out Fire', 
'disc_date_pre': 'Preliminary Discovery Date',
'disc_pre_year': 'Year of Preliminary Discovery', 
'disc_pre_month': 'Month of Preliminary Discovery', 
'wstation_usaf': 'Weather Station USAF', 'dstation_m': 'Distance to Station M',
'wstation_wban': 'Weather Station WBAN', 
'wstation_byear': 'Weather Station Begin Year', 
'wstation_eyear': 'Weather Station End Year',
'Vegetation': 'Dominant Vegetation',
'fire_mag': 'Magnitude of Fire',
'weather_file': 'Weather File', 
'Temp_pre_30': 'Temperature 30 Days Prior', 
'Temp_pre_15': 'Temperature 15 Days Prior', 
'Temp_pre_7': 'Temperature 7 Days Prior',
'Temp_cont': 'Temperature on Containment Day', 
'Wind_pre_30': 'Wind 30 Days Prior', 
'Wind_pre_15': 'Wind 15 Days Prior',
'Wind_pre_7': 'Wind 7 Days Prior', 
'Wind_cont': 'Wind on Containment Day',
'Hum_pre_30': 'Humidity 30 Days Prior',
'Hum_pre_15': 'Humidity 15 Days Prior', 
'Hum_pre_7': 'Humidity 7 Days Prior', 
'Hum_cont': 'Humidity on Containment Day', 
'Prec_pre_30': 'Precipitation 30 Days Prior',
'Prec_pre_15': 'Precipitation 15 Days Prior', 
'Prec_pre_7': 'Precipitation 7 Days Prior', 
'Prec_cont': 'Precipitation on Containment Day', 
'remoteness': 'Remoteness'

Los tipos de datos de cada columna:
Unnamed: 0.1                          int64
Unnamed: 0                            int64
Name of Fire                         object
Size of Fire                        float64
Class of Fire Size (A-G)             object
Cause of Fire                        object
Latitude of Fire                    float64
Longitude of Fire                   float64
State of Fire                        object
Discovery Date                       object
Containment Date                     object
Month of Discovery                   object
Final Discovery Date                 object
Final Containment Date               object
Time to Put Out Fire                 object
Preliminary Discovery Date           object
Year of Preliminary Discovery         int64
Month of Preliminary Discovery       object
Weather Station USAF                 object
Distance to Station M               float64
Weather Station WBAN                  int64
Weather Station Begin Year            int64
Weather Station End Year              int64
Dominant Vegetation                   int64
Magnitude of Fire                   float64
Weather File                         object
Temperature 30 Days Prior           float64
Temperature 15 Days Prior           float64
Temperature 7 Days Prior            float64
Temperature on Containment Day      float64
Wind 30 Days Prior                  float64
Wind 15 Days Prior                  float64
Wind 7 Days Prior                   float64
Wind on Containment Day             float64
Humidity 30 Days Prior              float64
Humidity 15 Days Prior              float64
Humidity 7 Days Prior               float64
Humidity on Containment Day         float64
Precipitation 30 Days Prior         float64
Precipitation 15 Days Prior         float64
Precipitation 7 Days Prior          float64
Precipitation on Containment Day    float64
Remoteness                          float64
dtype: object

## Common mistakes to avoid
Errores y dificultades temporales y espaciales: 
  1. Ignorar el retraso en la notificación: confundir la fecha de descubrimiento (DISCOVERY_DATE) o la fecha de contención con la hora exacta de ignición distorsiona las correlaciones predictivas con el clima. 
  2. Confundir atributos nominales y espaciales: atributos como ESTADO (STATE) y CONDADO (COUNTY) en las tablas de informes federales suelen ser campos de texto ingresados ​​manualmente —y no derivados de superposiciones espaciales—, lo que introduce errores tipográficos o discrepancias con la LATITUD y LONGITUD. 
  3. Malinterpretar incendios de tamaño cero: tratar los incendios con cero acres quemados o datos faltantes en este campo (FIRE_SIZE) como ruido algorítmico, en lugar de como intervenciones de contención rápida o quemas prescritas, distorsiona las métricas de gravedad. 

Errores de modelado y manejo de datos: 
  1. Ignorar una asimetría extrema: no aplicar transformaciones logarítmicas o escalado robusto a variables objetivo (como el área quemada) provoca que los valores atípicos extremos dominen los modelos basados ​​en gradientes. 
  2. División aleatoria para entrenamiento y prueba: utilizar divisiones aleatorias en datos de series temporales de incendios forestales provoca una fuga de datos, ya que eventos futuros filtran patrones temporales en los conjuntos de entrenamiento; se debe utilizar siempre validación cruzada basada en el tiempo o agrupada. 
  3. Pasar por alto la ausencia de agencias informantes: tratar la falta de datos en el campo de agencia informante (NWCG_REPORTING_AGENCY) como algo aleatorio en lugar de sistémico (por ejemplo, diferencias de seguimiento entre agencias locales y federales) introduce un sesgo oculto de selección de muestra. 

Preparación del modelo y limitaciones —
 1. Valores faltantes: porcentaje de valores nulos en las fechas de contención o en las tablas de atributos complementarios. 
 2. Desequilibrio de clases: variables objetivo sesgadas (p. ej., incendios de gran magnitud frente a la contención típica de incendios pequeños). 
 3. Errores comunes: sobreajuste a agrupaciones geográficas de alta densidad o ignorar los sesgos de la agencia informante.

## Structured output format

Análisis Exploratorio de Datos (EDA):
 1. Tendencias temporales (gráficos de frecuencia interanual y distribución de la estacionalidad).
 2. Distribución del tamaño: histograma de FIRE_SIZE con escala logarítmica debido a la fuerte asimetría hacia los incendios de menor tamaño. 
 3. Análisis de causas: desglose de incidentes según su origen (causados ​​por humanos frente a causados ​​por rayos).


## References to your other .md files



## Repository structure

- `EDA_FWVegRemCombined.ipynb` / `Main_EDA_FWVegRemCombined.ipynb` — EDA on the combined fire/weather/vegetation
  CSV. The `EDA_*` notebook is the full worked notebook; `Main_EDA_*` is a thin/experimental entry point.
- `EDA_FpaFod20170508.ipynb` / `Main_EDA_FpaFod20170508.ipynb` — same pattern, for the SQLite FPA-FOD dataset.
- `Functions.ipynb` — shared helper (`diagnostico_dataframe(df)`) that prints shape, dtypes, describe, and
  summary stats for a DataFrame. Not imported programmatically (no `.py` module) — code is copy/pasted between
  notebooks, so changes to this logic must be manually re-applied everywhere it's duplicated.
- `TemplatesEDA/` — reference/starter notebooks (`eda_starter.ipynb`, a Kaggle EDA tutorial) that the two
  `EDA_*` notebooks were structured after. Both EDA notebooks follow the same numbered section outline
  (0 Fichero de datos → 1 Importar librerías → 2 Carga de datos → 3 Exploración: `.head()`, `.tail()`, columns,
  shape, `.info()`, `.describe()`, rename columns, dtypes, null check, drop duplicates, outlier detection via
  boxplots/scatter/Z-score). When extending EDA, follow this existing section numbering convention rather than
  inventing a new structure.
- `BI/` — standalone static HTML dashboard (`index.html`) built with Tailwind, Plotly.js, and PapaParse
  (all vendored locally as `.js` files, no CDN, no bundler). It loads `BI/FW_Veg_Rem_Combined.csv` client-side
  with PapaParse, derives `region`/`season`/`is_human_caused` fields in `cleanData()`, and renders charts
  (temperature, humidity/precipitation, seasonal heatmap, yearly trends, cause donut, size histogram, US choropleth
  map) via Plotly. Same `-1` = missing-value convention as the source CSV. Open `BI/index.html` directly in a
  browser (or serve the folder) to run it — no build step.
- `DataBase/`, `DataSets/` — raw/processed data files (see above).

## Working with the notebooks

- No `requirements.txt`/`environment.yml` exists. Notebooks assume `pandas`, `numpy`, `seaborn`, `matplotlib`,
  and `scipy` are available in the active Python kernel.
- Load data with paths relative to the repository root (e.g. `pd.read_csv(r'DataSets\FW_Veg_Rem_Combined.csv')`),
  matching how `EDA_FWVegRemCombined.ipynb` does it. Some notebooks (e.g. cells in `EDA_FpaFod20170508.ipynb`
  and `Main_EDA_FpaFod20170508.ipynb`) instead hardcode absolute Windows paths — some of these are stale
  (pointing at a differently-named parent folder) and will fail as-is. When adding or fixing data-loading
  cells, prefer relative paths over hardcoded absolute ones.
- Notebook markdown/comments and dashboard copy are written in Spanish; keep new analysis narrative consistent
  with that.
