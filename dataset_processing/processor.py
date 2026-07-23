import pandas as pd
import numpy as np
import xarray as xr
from dataset_building.aemet.config import START_DATE as S_AEMET, END_DATE as E_AEMET
from dataset_building.era5.config import START_YEAR as S_ERA5, END_YEAR as E_ERA5
from dataset_processing.config import PROVINCES_CLM, COLS_TO_DROP, STATION_SELECTION_PARAMS, LOWER_LIMIT_YEAR_AEMET
import dataset_processing.utils as utils
from datetime import datetime

# Limit time span to overlap
s = max(S_ERA5, S_AEMET.year)
e = min(E_ERA5, E_AEMET.year)

def reduce_aemet_df_to_clm(df, s_df):
    '''
    Returns weather and stations dataframe with years 2010 on and only CLM stations

    Parameters:
        df: Pandas Dataframe containing AEMET OpenData weather data
        s_df: Pandas Dataframe containing AEMET stations

    Returns:
        df_clm: same dataframe limited to Castilla-La Mancha data
        s_df_clm: same dataframe as s_df limited to Castilla-La Mancha stations
    '''
    # Limit dfs
    df_clm = df.loc[
        (df['anio'] >= s) & 
        (df['anio'] <= e) &
        (df['provincia'].isin(PROVINCES_CLM))
    ].copy()

    s_df_clm = s_df.loc[
        (s_df['provincia'].isin(PROVINCES_CLM))
    ]

    # Remove unwanted cols
    df_clm.drop(columns=COLS_TO_DROP, inplace=True)

    # Remove unused categories
    df_clm = df_clm.apply(lambda col: col.cat.remove_unused_categories() if col.dtype.name == 'category' else col)
    s_df_clm = s_df_clm.apply(lambda col: col.cat.remove_unused_categories() if col.dtype.name == 'category' else col)

    # Reset indexes
    df_clm.reset_index(drop=True, inplace=True)
    s_df_clm.reset_index(drop=True, inplace=True)

    return df_clm, s_df_clm


def adjust_pressures(df, celsius=True):
    '''
    Calculates mean pressure from min and max pressure (rough estimate, potential source of error),
    adjusts pressures to mean sea level

    Parameters:
        df: Pandas Dataframe containing AEMET OpenData weather data
    
    Returns:
        df: Same dataframe with mean pressure and sea level pressures added
    '''
    # Estimate mean pressure
    df = df.copy()
    df['presMedia'] = (df['presMax'] + df['presMin']) / 2

    t = df['tmed']
    h = df['altitud']
    rh = df['hrMedia']

    # Calculate mean sea level pressures
    df['mslp_media'] = utils.sea_level_pressure_reduction(df['presMedia'], t, h, rh, celsius).astype('float32')
    df['mslp_max'] = utils.sea_level_pressure_reduction(df['presMax'], t, h, rh, celsius).astype('float32')
    df['mslp_min'] = utils.sea_level_pressure_reduction(df['presMin'], t, h, rh, celsius).astype('float32')

    return df


def match_era5_units(df):
    '''
    Changes dataframe column units to match ERA5 units

    Parameters:
        df: pandas df with aemet data and units
    
    Returns:
        df: pandas df with aemet data and units changed to match era5
    '''
    df = df.copy()
    # Celsius to Kelvin
    for col in ['tmax', 'tmin', 'tmed']:
        df[col] = df[col] + 273.15

    # hPa to Pa
    for col in ['mslp_media', 'mslp_max', 'mslp_min', 'presMax', 'presMin']:
        df[col] = df[col] * 100

    # mm to m
    df['prec'] = df['prec'] / 1000

    return df


def _nearest_pointwise(ds, lats, lons, times):
    '''
    Nearest neighbor vectorized selection (latitude, longitude, valid_time).
 
    Calculates nearest index to each axis at once with get_indexer and collects
    values with vectorized pintwise isel.

    Parameters:
        ds: open xarray.Dataset
        lats, lons: 1-D query coordinate arrays
        times: 1-D query date array/index
 
    Returns:
        np.ndarray float64 of length len(lats) with collected values.
        Unindexed points (-1) return NaN.
    '''
    var = list(ds.data_vars)[0]
 
    lat_pos = ds.indexes['latitude'].get_indexer(np.asarray(lats), method='nearest')
    lon_pos = ds.indexes['longitude'].get_indexer(np.asarray(lons), method='nearest')
    time_pos = ds.indexes['valid_time'].get_indexer(pd.to_datetime(times), method='nearest')
 
    dim = 'points'
    vals = ds[var].isel(
        latitude=xr.DataArray(lat_pos, dims=dim),
        longitude=xr.DataArray(lon_pos, dims=dim),
        valid_time=xr.DataArray(time_pos, dims=dim),
    ).values
    vals = np.asarray(vals, dtype='float64')
 
    bad = (lat_pos == -1) | (lon_pos == -1) | (time_pos == -1)
    if bad.any():
        vals[bad] = np.nan
    return vals
 
 
def fill_missing_values(df, vars_dir='era5_grid_data/combined'):
    '''
    Fills missing values from AEMET df using ERA5 data
 
    Parameters:
        df: Pandas Dataframe containing AEMET OpenData data
        vars_dir: path to directoy containing ERA5 variable files
 
    Returns:
        df: Pandas Dataframe with missing values filled
    '''
 
    # SHORT NAMES (second element in dict value tuples) WERE NOT USED
    # INSTEAD list(ds.data_vars)[0] RETURNS THE SHORT NAME USED TO ACCESS THE NUMERICAL VALUE
    files = {
        'mslp_media': (
            'CLM_daily_mean_mean_sea_level_pressure_0_daily-mean.nc', 'msl'
        ),
        'mslp_max': (
            'CLM_daily_max_mean_sea_level_pressure_0_daily-max.nc', 'msl'
        ),
        'mslp_min': (
            'CLM_daily_min_mean_sea_level_pressure_0_daily-min.nc', 'msl'
        ),
        'racha': (
            'CLM_daily_max_10m_wind_gust_since_previous_post_processing_0_daily-max.nc',
            '10fg'
        ),
        'tmed': (
            'CLM_daily_mean_2m_temperature_0_daily-mean.nc', '2tm'
        ),
        'tmax': (
            'CLM_daily_max_maximum_2m_temperature_since_previous_post_processing_stream-oper_daily-max.nc',
            'mx2t'
        ),
        'tmin': (
            'CLM_daily_min_minimum_2m_temperature_since_previous_post_processing_stream-oper_daily-min.nc',
            'mn2t'
        ),
        'prec': (
            'CLM_daily_sum_precipitation.nc', 'tp'
        ),
        'velmedia': (
            ('CLM_daily_mean_10m_u_component_of_wind_stream-oper_daily-mean.nc', '10u'),
            ('CLM_daily_mean_10m_v_component_of_wind_0_daily-mean.nc', '10v')
        ),
        'hrMedia': (
            ('CLM_daily_mean_2m_temperature_0_daily-mean.nc', '2t'),
            ('CLM_daily_mean_2m_dewpoint_temperature_0_daily-mean.nc', '2d')
        ),
    }
 
    # Loads data into a dictionary of items (filename, data)
    era5 = {}
    for col, mapping in files.items():
        # Data spread in diff files
        if isinstance(mapping[0], tuple):
            for fn, var in mapping:
                if fn not in era5:
                    era5[fn] = xr.open_dataset(f"{vars_dir}/{fn}")
        # Data in single file
        else:
            fn, var = mapping
            if fn not in era5:
                era5[fn] = xr.open_dataset(f"{vars_dir}/{fn}")
 
    df = df.copy()
 
    # fecha como datetime una sola vez (antes se hacia pd.to_datetime por fila)
    fecha = pd.to_datetime(df['fecha'])
 
    simple_cols = [c for c in files if c not in ('velmedia', 'hrMedia')]
 
    # Simple NaN filling (vectorizado: una seleccion por variable, no por fila)
    for col in simple_cols:
        mask = df[col].isna()
        if not mask.any():
            continue
        fn, _ = files[col]
        sel = mask.to_numpy()
        try:
            vals = _nearest_pointwise(
                era5[fn],
                df.loc[mask, 'latitud'].values,
                df.loc[mask, 'longitud'].values,
                fecha[mask].values,
            )
        except (KeyError, IndexError, ValueError, TypeError) as e:
            print(e)
            continue
        good = ~np.isnan(vals)
        if good.any():
            idx = df.index[sel][good]
            df.loc[idx, col] = vals[good].astype(df[col].dtype)
 
    # Wind speed calculated from u and v components
    mask = df['velmedia'].isna()
    if mask.any():
        (f_u, _), (f_v, _) = files['velmedia']
        sel = mask.to_numpy()
        try:
            u = _nearest_pointwise(era5[f_u], df.loc[mask, 'latitud'].values,
                                   df.loc[mask, 'longitud'].values, fecha[mask].values)
            v = _nearest_pointwise(era5[f_v], df.loc[mask, 'latitud'].values,
                                   df.loc[mask, 'longitud'].values, fecha[mask].values)
            vals = np.sqrt(u**2 + v**2)
        except (KeyError, IndexError, ValueError, TypeError) as e:
            print(e)
            vals = None
        if vals is not None:
            good = ~np.isnan(vals)
            if good.any():
                idx = df.index[sel][good]
                df.loc[idx, 'velmedia'] = vals[good].astype(df['velmedia'].dtype)
 
    # Relative humidity calculated from dewpoint temperature and temperature
    mask = df['hrMedia'].isna()
    if mask.any():
        (f_t, _), (f_td, _) = files['hrMedia']
        sel = mask.to_numpy()
        try:
            t = _nearest_pointwise(era5[f_t], df.loc[mask, 'latitud'].values,
                                   df.loc[mask, 'longitud'].values, fecha[mask].values)
            td = _nearest_pointwise(era5[f_td], df.loc[mask, 'latitud'].values,
                                    df.loc[mask, 'longitud'].values, fecha[mask].values)

            vals = np.asarray(
                utils.calculate_relative_humidity(td, t, celsius=False),
                dtype='float64'
            )
        except (KeyError, IndexError, ValueError, TypeError) as e:
            print(e)
            vals = None
        if vals is not None:
            good = ~np.isnan(vals)
            if good.any():
                idx = df.index[sel][good]
                df.loc[idx, 'hrMedia'] = vals[good].astype(df['hrMedia'].dtype)
 
    # Close datasets
    for ds in era5.values():
        ds.close()
 
    return df
 
 
def fill_missing_days(df, s_df, vars_dir='era5_grid_data/combined', s=s, e=e):
    '''
    Fills missing days with ERA5 information
 
    Parameters:
        df: dataframe containing AEMET OpenData weather info
        s_df: dataframe containing AEMET OpenData station info
        vars_dir: path to ERA5 var files
        s: start year, default uses overlap between ERA5 and AEMET (max start)
        e: end year, delfault uses overlap between ERA5 and AEMET (min end)
 
    Returns:
        filled_df: dataframe containing ERA5 information of missing days
        missing_df: dataframe containing which days were missing
    '''
 
    s_df = s_df.copy()
    s_df[['latitud', 'longitud']] = s_df[['latitud', 'longitud']].astype('float32')

    # Prepare df containing missing day info
    missing = utils.build_missing_days_df(df, s, e)
    missing_df = missing.merge(
        s_df[['indicativo', 'latitud', 'longitud', 'nombre', 'provincia']],
        on='indicativo', how='left'
    )
 
    # No short name now
    files = {
        'mslp_media': ['CLM_daily_mean_mean_sea_level_pressure_0_daily-mean.nc'],
        'mslp_max':   ['CLM_daily_max_mean_sea_level_pressure_0_daily-max.nc'],
        'mslp_min':   ['CLM_daily_min_mean_sea_level_pressure_0_daily-min.nc'],
        'racha':      ['CLM_daily_max_10m_wind_gust_since_previous_post_processing_0_daily-max.nc'],
        'tmed':       ['CLM_daily_mean_2m_temperature_0_daily-mean.nc'],
        'tmax':       ['CLM_daily_max_maximum_2m_temperature_since_previous_post_processing_stream-oper_daily-max.nc'],
        'tmin':       ['CLM_daily_min_minimum_2m_temperature_since_previous_post_processing_stream-oper_daily-min.nc'],
        'prec':       ['CLM_daily_sum_precipitation.nc'],
        'velmedia':   [
            'CLM_daily_mean_10m_u_component_of_wind_stream-oper_daily-mean.nc',
            'CLM_daily_mean_10m_v_component_of_wind_0_daily-mean.nc'
        ],
        'hrMedia':    [
            'CLM_daily_mean_2m_temperature_0_daily-mean.nc',
            'CLM_daily_mean_2m_dewpoint_temperature_0_daily-mean.nc'
        ],
    }
 
    # Load datasets into dict
    era5_ds = {}
    for var_files in files.values():
        for fn in var_files:
            if fn not in era5_ds:
                era5_ds[fn] = xr.open_dataset(f"{vars_dir}/{fn}")
 
    # Coordinates and dates of all rows at once
    n = len(missing_df)
    lats = missing_df['latitud'].values
    lons = missing_df['longitud'].values
    fecha = pd.to_datetime(missing_df['fecha'])
    times = fecha.values
 
    filled_df = pd.DataFrame({
        'indicativo': missing_df['indicativo'].values,
        'fecha': times,
        'nombre': missing_df['nombre'].values,
        'provincia': missing_df['provincia'].values,
    })
 
    simple_vars = [v for v in files if v not in ('velmedia', 'hrMedia')]
 
    # Simple filling (one vectorized selection per variable)
    for var in simple_vars:
        fn = files[var][0]
        try:
            vals = _nearest_pointwise(era5_ds[fn], lats, lons, times)
        except (KeyError, IndexError, ValueError, TypeError) as e:
            print(e)
            vals = np.full(n, np.nan)
        filled_df[var] = vals
 
    # velmedia = sqrt(u^2 + v^2)
    try:
        fn_u, fn_v = files['velmedia']
        u = _nearest_pointwise(era5_ds[fn_u], lats, lons, times)
        v = _nearest_pointwise(era5_ds[fn_v], lats, lons, times)
        filled_df['velmedia'] = np.sqrt(u**2 + v**2)
    except (KeyError, IndexError, ValueError, TypeError) as e:
        print(e)
        filled_df['velmedia'] = np.full(n, np.nan)
 
    # hrMedia calculated from dewpoint temperature and temperature
    try:
        fn_t, fn_td = files['hrMedia']
        t = _nearest_pointwise(era5_ds[fn_t], lats, lons, times)
        td = _nearest_pointwise(era5_ds[fn_td], lats, lons, times)
        filled_df['hrMedia'] = np.asarray(
            utils.calculate_relative_humidity(td, t, celsius=False),
            dtype='float64'
        )
    except (KeyError, IndexError, ValueError, TypeError) as e:
        print(e)
        filled_df['hrMedia'] = np.full(n, np.nan)
 
    # Close datasets
    for ds in era5_ds.values():
        ds.close()
 
    return filled_df, missing_df


def get_full_era5_df(df, s_df, vars_dir='era5_grid_data/combined', s=s, e=e):
    '''
    Returns full ERA5 data from start year to end dear, for comparing with AEMET data.
 
    Parameters:
        df: dataframe containing AEMET OpenData weather info
        s_df: dataframe containing AEMET OpenData station info
        vars_dir: path to ERA5 var files
        s: start year, default uses overlap between ERA5 and AEMET (max start)
        e: end year, delfault uses overlap between ERA5 and AEMET (min end)
 
    Returns:
        full_df: dataframe containing ERA5 information of all days from s to e
    '''
    s_df = s_df.copy()
    s_df[['latitud', 'longitud']] = s_df[['latitud', 'longitud']].astype('float32')

    # Prepare df containing missing day info
    stations = df['indicativo'].unique()
    dates = pd.date_range(datetime(s, 1, 1), datetime(e, 12, 31), freq='D')
    df_idx = pd.MultiIndex.from_product([stations, dates], names=['indicativo', 'fecha']).to_frame(index=False)
    full_df = df_idx.merge(
        s_df[['indicativo', 'latitud', 'longitud', 'nombre', 'provincia']],
        on='indicativo', how='left'
    )

    files = {
        'mslp_media': ['CLM_daily_mean_mean_sea_level_pressure_0_daily-mean.nc'],
        'mslp_max':   ['CLM_daily_max_mean_sea_level_pressure_0_daily-max.nc'],
        'mslp_min':   ['CLM_daily_min_mean_sea_level_pressure_0_daily-min.nc'],
        'racha':      ['CLM_daily_max_10m_wind_gust_since_previous_post_processing_0_daily-max.nc'],
        'tmed':       ['CLM_daily_mean_2m_temperature_0_daily-mean.nc'],
        'tmax':       ['CLM_daily_max_maximum_2m_temperature_since_previous_post_processing_stream-oper_daily-max.nc'],
        'tmin':       ['CLM_daily_min_minimum_2m_temperature_since_previous_post_processing_stream-oper_daily-min.nc'],
        'prec':       ['CLM_daily_sum_precipitation.nc'],
        'velmedia':   [
            'CLM_daily_mean_10m_u_component_of_wind_stream-oper_daily-mean.nc',
            'CLM_daily_mean_10m_v_component_of_wind_0_daily-mean.nc'
        ],
        'hrMedia':    [
            'CLM_daily_mean_2m_temperature_0_daily-mean.nc',
            'CLM_daily_mean_2m_dewpoint_temperature_0_daily-mean.nc'
        ],
    }

    # Load datasets into dict
    era5_ds = {}
    for var_files in files.values():
        for fn in var_files:
            if fn not in era5_ds:
                era5_ds[fn] = xr.open_dataset(f"{vars_dir}/{fn}")
 
    # Coordinates and dates of all rows at once
    n = len(full_df)
    lats = full_df['latitud'].values
    lons = full_df['longitud'].values
    fecha = pd.to_datetime(full_df['fecha'])
    times = fecha.values
 
    full_df = pd.DataFrame({
        'indicativo': full_df['indicativo'].values,
        'fecha': times,
        'nombre': full_df['nombre'].values,
        'provincia': full_df['provincia'].values,
    })
 
    simple_vars = [v for v in files if v not in ('velmedia', 'hrMedia')]
 
    # Simple filling (one vectorized selection per variable)
    for var in simple_vars:
        fn = files[var][0]
        try:
            vals = _nearest_pointwise(era5_ds[fn], lats, lons, times)
        except (KeyError, IndexError, ValueError, TypeError) as e:
            print(e)
            vals = np.full(n, np.nan)
        full_df[var] = vals
 
    # velmedia = sqrt(u^2 + v^2)
    try:
        fn_u, fn_v = files['velmedia']
        u = _nearest_pointwise(era5_ds[fn_u], lats, lons, times)
        v = _nearest_pointwise(era5_ds[fn_v], lats, lons, times)
        full_df['velmedia'] = np.sqrt(u**2 + v**2)
    except (KeyError, IndexError, ValueError, TypeError) as e:
        print(e)
        full_df['velmedia'] = np.full(n, np.nan)
 
    # hrMedia calculated from dewpoint temperature and temperature
    try:
        fn_t, fn_td = files['hrMedia']
        t = _nearest_pointwise(era5_ds[fn_t], lats, lons, times)
        td = _nearest_pointwise(era5_ds[fn_td], lats, lons, times)
        full_df['hrMedia'] = np.asarray(
            utils.calculate_relative_humidity(td, t, celsius=False),
            dtype='float64'
        )
    except (KeyError, IndexError, ValueError, TypeError) as e:
        print(e)
        full_df['hrMedia'] = np.full(n, np.nan)
 
    # Close datasets
    for ds in era5_ds.values():
        ds.close()
 
    return full_df


def select_aemet_stations(df, s_df):
    '''
    Select AEMET stations according to params set in config

    Parameters:
        df: dataframe containing AEMET OpenData weather info
        s_df: dataframe containing AEMET OpenData station info

    Returns:
        curated_df: dataframe containing AEMET OpenData weather info from selected stations
        curated_s_df: dataframe containing AEMET OpenData station info from selected stations
    '''
    df = df.copy().drop(columns=COLS_TO_DROP)
    df = df[df['anio'] >= LOWER_LIMIT_YEAR_AEMET]
    s_df = s_df.copy()

    nan_by_id = df.groupby('indicativo', observed=True).apply(lambda x: x.isna().mean(), include_groups=False) \
    .drop(columns=['anio', 'mes', 'dia', 'diaAnio', 'altitud', 'longitud', 'latitud', 'fecha', 'nombre', 'provincia'])

    threshold_cols = STATION_SELECTION_PARAMS['threshold_cols']
    threshold = STATION_SELECTION_PARAMS['nan_threshold']

    mask = (nan_by_id[threshold_cols] <= threshold).all(axis=1) 
    selected_by_id = nan_by_id[mask] # Filters rows with col <= threshold for each column defined in params

    s_df = s_df.loc[
        (s_df['proporcion_dias_perdidos'] <= STATION_SELECTION_PARAMS.get('missing_days', 0.01))
        & (s_df['fecha_min'] <= STATION_SELECTION_PARAMS.get('min_date', '2010-01-01'))
        & (s_df['fecha_max'] >= STATION_SELECTION_PARAMS.get('min_coverage', '2024-01-01'))
        & (s_df['altitud'] <= STATION_SELECTION_PARAMS.get('max_altitude', 750))
    ]

    # print(mask)
    # print(selected_by_id.info())
    # print(s_df.info())

    curated_s_df = s_df.merge(selected_by_id, on='indicativo')
    curated_df = df.loc[df['indicativo'].isin(curated_s_df['indicativo'])].reset_index(drop=True)
    return curated_df, curated_s_df

    