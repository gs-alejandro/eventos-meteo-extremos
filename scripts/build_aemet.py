"""
EL PROPÓSITO DE ESTE ARCHIVO ES CONSTRUIR ARCHIVOS .csv DE DATOS METEOROLÓGICOS PROPORCIONADOS POR AEMET OPENDATA
THE PURPOSE OF THIS FILE IS BUILDING .csv FILES CONTAINING METEOROLOGICAL DATA FROM AEMET OPENDATA
"""

import pandas as pd 
import os
import logging      # keeping track of correct functioning
from dataset_building.aemet.config import START_DATE, END_DATE
import dataset_building.aemet.builder as builder

# SET UP LOG CONFIG
logging.basicConfig(
    filename='AEMET_build_events.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

'''
INFO ABOUT RELEVANT API CALLS:

LIST OF STATIONS
/api/valores/climatologicos/inventarioestaciones/estaciones/{estaciones}

(Maximum time span is 6 months and maximum stations is 25)
CLIMATOLOGIC VALUES DAILY IN GIVEN STATIONS BETWEEN GIVEN DATES
/api/valores/climatologicos/diarios/datos/fechaini/{fechaIniStr}/fechafin/{fechaFinStr}/estacion/{idema}

(Maximum time span is 15 days)
CLIMATOLOGIC VALUES DAILY IN ALL STATIONS BETWEEN GIVEN DATES
/api/valores/climatologicos/diarios/datos/fechaini/{fechaIniStr}/fechafin/{fechaFinStr}/todasestaciones

AVERAGE MONTHLY AND YEARLY CLIMATOLOGIC VALUES IN A GIVEN STATION AND YEARS
/api/valores/climatologicos/mensualesanuales/datos/anioini/{anioIniStr}/aniofin/{anioFinStr}/estacion/{idema}

NORMAL CLIMATOLOGIC VALUES FOR GIVEN STATION FROM 1991 TO 2020
/api/valores/climatologicos/normales/estacion/{idema}

EXTREME VALUES FOR THE STATION AND THE VARIABLE
/api/valores/climatologicos/valoresextremos/parametro/{parametro}/estacion/{idema}
    param vals:
        V - Wind
        T - Temperature
        P - Precipitation
'''

if __name__ == '__main__':

    # Make sure directories exist
    os.makedirs('datasets/csv', exist_ok=True)
    os.makedirs('datasets/parquet', exist_ok=True)

    # Get weather station info and save it
    print('Fetching station data')
    station_info = builder.get_station_info()
    station_df = pd.DataFrame(station_info)
    print('Saving station data')
    station_df.to_csv('datasets/csv/RAW_AEMET_stations.csv', index=False)

    # Remove unwanted columns
    station_df.drop(columns=['provincia', 'altitud', 'nombre', 'indsinop'], inplace=True)

    output_file = f'datasets/csv/RAW_AEMET_{START_DATE.strftime('%Y-%m-%d')}_{END_DATE.strftime('%Y-%m-%d')}.csv'

    # Get weather data and add coordinates, order by date and station
    builder.fetch_aemet_data(output_file=output_file)

    weather_df = pd.read_csv(f'datasets/csv/RAW_AEMET_{START_DATE.strftime('%Y-%m-%d')}_{END_DATE.strftime('%Y-%m-%d')}.csv', low_memory=False)

    print('Adding coordinates to weather data')
    weather_df = weather_df.merge(station_df, on='indicativo', how='left')
    weather_df = weather_df.sort_values(by=['fecha', 'indicativo'], ascending=[True, True]).reset_index(drop=True)

    # Save dataframe in .csv file
    print('Saving weather data with coordinates')
    weather_df.to_csv(f'datasets/csv/RAW_AEMET_{START_DATE.strftime('%Y-%m-%d')}_{END_DATE.strftime('%Y-%m-%d')}_COOR.csv', index=False)

    # print('PROCESS COMPLETE.')
    print('Result:')
    print(weather_df.info())
