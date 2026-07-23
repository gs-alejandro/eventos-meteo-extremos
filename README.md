# Predicción de eventos meteorológicos extremos

Proyecto dedicado a entrenamiento de modelos de Machine Learning para predicción de eventos meteorológicos extremos. 

Dependencias instalables mediante el comando `pip install -r requirements.txt`

## Pasos para la recreación de los conjuntos de datos (opcional)

### **Estos pasos son solo necesarios si se desea reconstruir los datos.**

### 0. Cambiar configuración (opcional).

Si se quiere modificar la configuración de datos recogidos, para cambiar los periodos de tiempo basta con cambiar:
- `START_DATE` y `END_DATE` en el archivo [`dataset_building/aemet/config.py`](dataset_building/aemet/config.py).
- `START_YEAR` y `END_YEAR` en el archivo [`dataset_building/era5/config.py`](dataset_building/era5/config.py).

Si además se quiere cambiar el comportamiento del procesamiento en el archivo [`dataset_processing/config.py`](dataset_processing/config.py):
- `LOWER_LIMIT_YEAR_AEMET` para cambiar el límite inferior aplicado posteriormente a los datos de AEMET.
- `STATION_SELECTION_PARAMS` para cambiar los criterios de selección de estaciones para el conjunto de datos de España.
- `PROVINCES_CLM` para definir las provincias que se usan para el conjunto de datos de CLM.
- `COLS_TO_DROP` para definir las columnas que se eliminan en el procesamiento.

### 1. Obtener una llave de API de [AEMET OpenData](https://opendata.aemet.es/centrodedescargas/inicio).

Primero, hay que solicitar una llave de API de [AEMET OpenData](https://opendata.aemet.es/centrodedescargas/inicio), introduciendo el correo electónico y siguiendo los pasos.

Con esta llave de API, se debe crear un archivo `.env` que contenga: AEMET_API_KEY=[LLAVE]

### 2. Preparar [`cdsapi`](https://cds.climate.copernicus.eu/how-to-api).

En la *Climate Data Store* ya existe una [guía](https://cds.climate.copernicus.eu/how-to-api) de cómo hacer el *setup*. 

Adicionalmente, yo tuve que aceptar la licencia del conjunto de datos [ERA5 daily aggregated data on single levels from 1940 to present](https://doi.org/10.6084/m9.figshare.21789074).

### 3. Ejecutar [`execute.py`](execute.py) para lanzar todos los scripts de recolección y procesamiento de datos.

## Pasos para el entrenamiento de los modelos

### 0. Cambiar configuración (opcional).

Se puede modificar el diccionario `CONFIG` de [`analysis_modeling/config.py`](analysis_modeling/config.py) para modificar:

- Rutas origen de los datos
- Rutas destino de los conjuntos de datos de entrenamiento
- Rutas destino de los registros y modelos resultantes del entrenamiento
- Columnas y periodos de tiempo de las agregaciones temporales 

### 1. Ejecutar la libreta [`analysis_modeling/4_feature_engineering.ipynb`](analysis_modeling/4_feature_engineering.ipynb) para crear los conjuntos de entrenamiento.

### 2. Ejecutar la libreta [`analysis_modeling/5_modeling_evaluation.ipynb`](analysis_modeling/5_modeling_evaluation.ipynb) para entrenar los modelos

### 3. Opcionalmente, ejecutar la libreta [`analysis_modeling/6_evaluation.ipynb`](analysis_modeling/6_evaluation.ipynb) para mostrar gráficos del rendimiento de los modelos en el *split* de prueba.




# Datos utilizados

## Datos de AEMET

- Este proyecto utiliza datos proporcionados por la API **AEMET OpenData**
- Fuente: Agencia Estatal de Meteorología (AEMET)
- Licencia: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Fecha de acceso aproximada: 2025

> Los datos han sido procesados y/o modificados, por lo que **AEMET no se hace responsable** del resultado final.

## Datos de ERA5

- Este proyecto utiliza datos proporcionados por **Copernicus Climate Data Store**
- Fuente: Copernicus Climate Change Service (C3S): Climate Data Store (CDS)
- Licencia: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Dataset: ERA5 daily aggregated data on single levels from 1940 to present
- DOI: [10.24381/cds.4991cf48](https://doi.org/10.24381/cds.4991cf48)
- Fecha de acceso aproximada: 2025

> Los datos han sido procesados y/o modificados en este proyecto, por lo que **C3S no se hace responsable** del resultado final.

> **Attribution:** Generated using Copernicus Climate Change Service information (2025).  
> Neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data it contains.


## Mapas climáticos Köppen-Geiger 
- Este proyecto usa datos de mapas Köppen-Geiger proporcionados por **Beck et al.** a través de **Figshare**
- Fuente: Figshare
- Licencia: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Archivo: `koppen_geiger_tif.zip`
- DOI: [10.6084/m9.figshare.21789074](https://doi.org/10.6084/m9.figshare.21789074)
- Fecha de acceso aproximado: 2026

> Los datos han sido procesados y/o modificados en este proyecto, por lo que **Beck et al. / Figshare no se hacen responsables** del resultado final.
