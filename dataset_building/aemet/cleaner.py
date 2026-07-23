import pandas as pd 
import numpy as np  
import gc


def dms_to_decimal(coordinate):
    '''
    Function to convert DMS (grado, minuto, segundo) coordinates to decimal degrees
    '''
    # Separate DMS coordinate
    degrees = pd.to_numeric(coordinate.str.slice(0, 2), errors='coerce')
    minutes = pd.to_numeric(coordinate.str.slice(2, 4), errors='coerce')
    seconds = pd.to_numeric(coordinate.str.slice(4, 6), errors='coerce')
    direction = coordinate.str.slice(6, 7)

    # Calculate decimal degrees
    decimal = degrees + minutes/60 + seconds/3600
    
    # S and W coordinates will be negative
    decimal = decimal.where(direction.isin(['N', 'E']), -decimal)

    return decimal


def fix_aemet_dtypes(in_df):
    '''
    Function to fix datatypes of a given AEMET dataset

    Args:
        df: pandas dataframe with all columns as returned by AEMET

    Returns:
        pandas dataframe with correct column dtype
    '''
    df = in_df.copy()

    # CATEGORICAL COLUMNS
    df['indicativo'] = df['indicativo'].astype('category')
    df['nombre'] = df['nombre'].astype('category')
    df['provincia'] = df['provincia'].astype('category')

    # DATES
    df['fecha'] = pd.to_datetime(df['fecha'])
    df['anio'] = df['fecha'].dt.year
    df['mes'] = df['fecha'].dt.month
    df['dia'] = df['fecha'].dt.day
    df['diaAnio'] = df['fecha'].dt.dayofyear

    # NUMERIC COLUMNS STORED AS STRING THAT CAN BE EASILY CONVERTED TO NUMERIC
    numeric_cols = [
        'tmax', 'tmin', 'tmed', # TEMPERATURES
        'velmedia', 'racha',    # WIND
        'sol',                  # SUN EXPOSURE
        'presMax', 'presMin',   # PRESSURE
    ]

    for col in numeric_cols:
        df[col] = df[col].str.replace(',', '.', regex=False) 
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # PRECIPITATION
    df['prec'] = df['prec'].str.strip().str.lower()
    # Add columns for representing acum and ip vals
    df['precAcum'] = np.where(df['prec'] == 'acum', 1, 0)
    df['precIp'] = np.where(df['prec'] == 'ip', 1, 0)
    df.loc[df['prec'] == 'ip', 'prec'] = 0 # Turn ip to 0
    df['prec'] = df['prec'].str.replace(',', '.', regex=False)
    df['prec'] = pd.to_numeric(df['prec'], errors='coerce')

    # HOURS OF OCCURRENCE
    hour_cols = ['horatmax', 'horatmin', 'horaracha', 'horaHrMax', 'horaHrMin'] 
    for col in hour_cols:
        # Flag various occurrences and convert to minutes
        flag_col = f'{col}Varias'
        df[flag_col] = (df[col] == 'Varias').astype(int)
        # Minutes
        dt = pd.to_datetime(df[col], format='%H:%M', errors='coerce')
        df[col] = dt.dt.hour * 60 + dt.dt.minute

    # These cols have precision up to an hour, convert to minutes
    hour_cols = ['horaPresMax', 'horaPresMin']
    for col in hour_cols:
        flag_col = f'{col}Varias'
        df[flag_col] = (df[col] == 'Varias').astype(int)
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col] * 60

    # COORDINATES
    df['latitud'] = df['latitud'].astype(str)
    df['longitud'] = df['longitud'].astype(str)
    df['latitud'] = dms_to_decimal(df['latitud'])
    df['longitud'] = dms_to_decimal(df['longitud'])

    # FLAG DIR COL
    df.loc[df['dir'] == 88, 'dir'] = pd.NA  # 88 = no data
    df['dirVarias'] = (df['dir'] == 99)     # 99 = variable direction
    df.loc[df['dir'] == 99, 'dir'] = pd.NA

    new_order = [
        # DATOS DE REGISTRO
        'fecha',        # Fecha de registro / Register date
        'indicativo',   # Identificador de estación meteorológica 
        'nombre',       # Nombre de estación meteorológica
        'provincia',    # Provincia

        # DATOS DE TEMPERATURA
        'tmed',         # Temperatura media del día
        'tmax',         # Temperatura máxima del día
        'tmin',         # Temperatura mínima del día
        'horatmax',     # Hora de temperatura máxima
        'horatmin',     # Hora de temperatura mínima
        'horatmaxVarias',
        'horatminVarias',

        # DATOS DE VIENTO
        'velmedia',     # Velocidad media del viento
        'racha',        # Velocidad máxima de viento
        'horaracha',    # Hora de mayor racha de viento
        'dir',          # Dirección de mayor racha de viento
        'dirVarias',    # Varias direcciones

        # DATOS DE HUMEDAD
        'hrMedia',      # Humedad media
        'hrMax',        # Humedad máxima
        'hrMin',        # Humedad mínima
        'horaHrMax',    # Hora de humedad máxima
        'horaHrMin',    # Hora de humedad mínima
        'horaHrMaxVarias',  
        'horaHrMinVarias',  

        # DATOS DE PRECIPITACIONES
        'prec',         # Cantidad de precipitación
        'precIp',       # Flag de Ip (inferior a 0.1mm)
        'precAcum',     # Flag de Acum (precipitación acumulada)

        # DATOS DE INSOLACIÓN
        'sol',          # Horas de insolación

        # DATOS DE PRESIÓN
        'presMax',      # Presión máxima
        'presMin',      # Presión mínima
        'horaPresMax',  # Hora de presión máxima
        'horaPresMin',  # Hora de presión mínima
        'horaPresMaxVarias', 
        'horaPresMinVarias', 

        # DATOS DE FECHA
        'anio',         # Año del registro
        'mes',          # Mes del año del registro
        'dia',          # Día del mes del registro
        'diaAnio',      # Día del año del registro

        # DATOS DE UBICACIÓN
        'altitud',
        'longitud',
        'latitud'
    ]

    # print(df.info())

    df = optimize_datatypes(df)

    df = df[new_order]

    return df


def add_completeness_info(w_df, s_df):

    w_df = w_df.copy()
    # Cast to categorical to reduce memory used for IDs
    w_df['indicativo'] = w_df['indicativo'].astype('category')

    # Process by station to avoid excessive RAM usage
    station_info = [] 
    for uid, group in w_df.groupby('indicativo', observed=True):
        min_date = group['fecha'].min()
        max_date = group['fecha'].max()

        # Calculate missing days
        full_days_count = (max_date - min_date).days + 1
        actual_days_count = len(group)
        missing_days = full_days_count - actual_days_count

        # Calculate missing values
        nan_dict = {}
        for col in w_df.columns:
            nans = group[col].isna().sum()
            nan_dict[f'nan_{col}'] = nans

        completeness_dict = {
            'indicativo': uid,
            'n_dias_reales': actual_days_count,
            'n_dias_teoricos': full_days_count,
            'n_dias_perdidos': missing_days,
            'fecha_min': min_date,
            'fecha_max': max_date,
            'proporcion_dias_perdidos': missing_days / full_days_count if full_days_count > 0 else 0               
        }

        completeness_dict.update(nan_dict)

        station_info.append(completeness_dict)

    gap_counts = pd.DataFrame(station_info)

    stations = s_df
    stations['latitud'] = dms_to_decimal(stations['latitud'].astype(str))
    stations['longitud'] = dms_to_decimal(stations['longitud'].astype(str))

    # Return station dataframe with completeness info
    return s_df.merge(gap_counts, on='indicativo', how='left')


def optimize_datatypes(df):
    flag_cols = [
        'dirVarias',
        'horaHrMaxVarias', 
        'horaHrMinVarias', 
        'precIp',
        'precAcum',
        'horaPresMaxVarias',
        'horaPresMinVarias',
    ]

    int_cols = df.select_dtypes('integer').columns
    float_cols = df.select_dtypes('float').columns

    for col in int_cols:
        df[col] = pd.to_numeric(df[col], downcast='integer')

    for col in float_cols:
        df[col] = pd.to_numeric(df[col], downcast='float')

    for col in flag_cols:
        if col in df.columns:
            df[col] = df[col].astype('int8')

    return df