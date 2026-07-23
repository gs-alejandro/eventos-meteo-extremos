import pandas as pd
import gc
from dataset_building.aemet.config import START_DATE, END_DATE
from dataset_building.aemet.cleaner import fix_aemet_dtypes, add_completeness_info
from tqdm import tqdm

if __name__ == '__main__':

    # Read dataframe in .csv file
    file = f'datasets/csv/RAW_AEMET_{START_DATE.strftime('%Y-%m-%d')}_{END_DATE.strftime('%Y-%m-%d')}_COOR.csv'
    out_file = f'datasets/parquet/AEMET_{START_DATE.strftime('%Y-%m-%d')}_{END_DATE.strftime('%Y-%m-%d')}.parquet'

    print('Reading weather data')
    weather_df = pd.read_csv(file, engine='pyarrow')

    # Clean datatypes
    print('Cleaning weather data')
    weather_df = fix_aemet_dtypes(weather_df)

    # Save clean parquet
    print('Saving as parquet')
    weather_df.to_parquet(out_file, index=False)

    stations_df = pd.read_csv('datasets/csv/RAW_AEMET_stations.csv', engine='pyarrow')
    # Add completeness info to station data
    print('Adding completeness information to station dataframe')
    stations_df = add_completeness_info(weather_df, stations_df)

    print('Saving new information to csv')
    stations_df.to_csv('datasets/csv/AEMET_stations.csv', index=False)

    print('Result:')
    print(weather_df.info())
