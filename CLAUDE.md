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
• Entorno de desarrollo: Jupyter Notebooks / Google Colab (aprovechando 
recursos en la nube).
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
- **`DataBase/FPA_FOD_20170508.sqlite`** — the raw FPA-FOD wildfire spatial database (`Fires` table plus
  SpatiaLite geometry tables). This file is **git-ignored** (too large to commit) and must exist locally
  for the `EDA_FpaFod20170508.ipynb` notebook to run.
- `DataSets/Wildfire_Weather_Merged_new.csv`, `DataSets/acres.csv`, `DataSets/fire-occurence.csv`,
  `DataSets/fires.csv` — supplementary/intermediate CSVs referenced by exploratory work.

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
