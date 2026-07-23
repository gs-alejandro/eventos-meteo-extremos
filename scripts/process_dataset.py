import pandas as pd
import os
import gc
import dataset_processing.processor as processor
import dataset_building.aemet.cleaner as cleaner
from dataset_building.aemet.config import START_DATE, END_DATE
from tqdm import tqdm

if __name__ == '__main__':

    df_file = f'datasets/parquet/AEMET_{START_DATE.strftime('%Y-%m-%d')}_{END_DATE.strftime('%Y-%m-%d')}.parquet'
    s_df_file = 'datasets/csv/AEMET_stations.csv'
    df = pd.read_parquet(df_file, engine='pyarrow')
    s_df = pd.read_csv(s_df_file)

    print('Selecting stations based on processing config')
    curated_df, curated_s_df = processor.select_aemet_stations(df, s_df)
    
    # print(curated_df.loc[(curated_df['nombre'] == 'MADRID AEROPUERTO'),['indicativo', 'nombre', 'provincia']].drop_duplicates())

    print('Calculating sea level pressure')
    curated_df = processor.adjust_pressures(curated_df)

    # print(curated_df.loc[(curated_df['nombre'] == 'MADRID AEROPUERTO'),['indicativo', 'nombre', 'provincia']].drop_duplicates())

    print('Renaming provinces with various names to single name')
    curated_df['provincia'] = (
        curated_df['provincia']
        .astype('string')
        .replace({
            'STA. CRUZ DE TENERIFE': 'SANTA CRUZ DE TENERIFE',
            'ILLES BALEARS': 'BALEARES'
        })
        .astype('category')
    )

    # print(curated_df.loc[(curated_df['nombre'] == 'MADRID AEROPUERTO'),['indicativo', 'nombre', 'provincia']].drop_duplicates())

    print('Saving Spain datasets')
    curated_df.to_parquet(f'datasets/parquet/CURATED_AEMET.parquet', index=False)
    curated_s_df.to_csv(f'datasets/csv/CURATED_AEMET_stations.csv', index=False)

    os.makedirs('datasets_clm', exist_ok=True)
    os.makedirs('datasets_clm/main', exist_ok=True)
    os.makedirs('datasets_clm/aux', exist_ok=True)

    # This is for experiment with supplementary data source for CLM only
    print('CLM dataset creation: reducing scope + adding sea level')
    df, s_df = processor.reduce_aemet_df_to_clm(df, s_df)
    df = processor.adjust_pressures(df)
    s_df.to_csv('datasets_clm/aux/AEMET_stations.csv', index=False)

    print('CLM dataset creation: getting full ERA5 dataset')
    full_df = processor.get_full_era5_df(df, s_df)
    full_df.to_parquet('datasets_clm/main/ERA5_daily.parquet', index=False)
    
    # Match units and vars
    print('CLM dataset creation: matching units and vars')
    df = processor.match_era5_units(df)
    print('CLM dataset creation: filling missing values in existing registers')
    df = processor.fill_missing_values(df)
    df = df.drop(columns=['presMax', 'presMin', 'presMedia'])
    df = cleaner.optimize_datatypes(df)
    df.to_parquet('datasets_clm/aux/AEMET_NAN_FILLED_daily_weather.parquet', index=False)

    print('CLM dataset creation: filling missing registers with ERA5 data')
    filled_df, missing_df = processor.fill_missing_days(df, s_df)
    filled_df = cleaner.optimize_datatypes(filled_df)
    filled_df = filled_df.merge(s_df[['indicativo', 'latitud', 'longitud', 'altitud']], on='indicativo', how='left')
    filled_df['anio'] = filled_df['fecha'].dt.year
    filled_df['mes'] = filled_df['fecha'].dt.month
    filled_df['dia'] = filled_df['fecha'].dt.day
    filled_df['diaAnio'] = filled_df['fecha'].dt.dayofyear
    df['source'] = 'AEMET'
    filled_df['source'] = 'ERA5'
    final_df = pd.concat([df, filled_df], ignore_index=True)
    final_df[['latitud', 'longitud']] = final_df[['latitud', 'longitud']].astype('float32')
    final_df['altitud'] = final_df['altitud'].astype('int32')

    print('CLM dataaset creation: saving ')
    final_df.to_parquet('datasets_clm/main/AEMET_ERA5_daily.parquet', index=False)
    missing_df.to_parquet('datasets_clm/aux/AEMET_missing_days.parquet', index=False)
    
    # print(final_df.info())

