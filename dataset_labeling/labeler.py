import numpy as np
import pandas as pd
from dataset_labeling.config import *

def optimize_datatypes(df):
    int_cols = df.select_dtypes('integer').columns
    float_cols = df.select_dtypes('float').columns

    for col in int_cols:
        df[col] = pd.to_numeric(df[col], downcast='integer')

    for col in float_cols:
        df[col] = pd.to_numeric(df[col], downcast='float')

    return df

def build_daily_thresholds(df, window=2, baseline_year=2018):
    # Global thresholds per station
    baseline = df[df['anio'] <= baseline_year]
    g = baseline.groupby('indicativo', observed=True)
    global_thresholds = pd.DataFrame({
        'global_tmax_threshold':     g['tmax'].quantile(GLOBAL_TMAX_PERCENTILE),
        'global_tmin_threshold':     g['tmin'].quantile(GLOBAL_TMIN_PERCENTILE),
        'global_prec_threshold':     g['prec'].quantile(GLOBAL_PREC_PERCENTILE),
        'global_velmedia_threshold': g['velmedia'].quantile(GLOBAL_WIND_PERCENTILE),
        'global_racha_threshold':    g['racha'].quantile(GLOBAL_WIND_PERCENTILE),
    })

    offsets = np.arange(-window, window + 1)    # -window .. +window

    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Index.repeat.html
    # Repeats as (1, 1, 2, 2) not as (1, 2, 1, 2), each register is repeated 2window + 1 times
    repeated = baseline.loc[baseline.index.repeat(len(offsets))]
    tiled = np.tile(offsets, len(baseline)) # tiene el offset
    
    """
    For each row, c_month and c_day col indicates which window center day of the year it belongs to.
    Simplified example rows to explain with window = 1:
        indicativo  fecha       tmax    c_month c_day
        example     2018-01-01  11      12      31
        example     2018-01-01  11      1       1
        example     2018-01-01  11      1       2
        example     2018-01-02  9       1       1
        example     2018-01-02  9       1       2
        example     2018-01-02  9       1       3
        example     2018-01-03  7       1       2
        example     2018-01-03  7       1       3
        example     2018-01-03  7       1       4
    """
    center_date = repeated['fecha'].to_numpy() + tiled.astype('timedelta64[D]')
    center_date = pd.DatetimeIndex(center_date)
    repeated = repeated.assign(c_month=center_date.month, c_day=center_date.day) 

    # Then group by center column to get local calculations
    grp = repeated.groupby(['indicativo', 'c_month', 'c_day'], observed=True)
    local_thresholds = pd.DataFrame({
        'tmax_threshold':       grp['tmax'].quantile(EXTREME_TMAX_PERCENTILE),
        'tmin_threshold':       grp['tmin'].quantile(EXTREME_TMIN_PERCENTILE),
        'prec_threshold':       grp['prec'].quantile(EXTREME_PREC_PERCENTILE),
        'velmedia_threshold':   grp['velmedia'].quantile(EXTREME_WIND_PERCENTILE),
        'racha_threshold':      grp['racha'].quantile(EXTREME_WIND_PERCENTILE),
    }).reset_index()

    # Merge with global calculations
    return local_thresholds.merge(global_thresholds, on='indicativo', how='left')


def label_rows(df, thresholds):
    df = df.copy()
    df['c_month'] = df['mes']
    df['c_day']   = df['dia']

    l = df.merge(thresholds, on=['indicativo', 'c_month', 'c_day'], how='left')

    def flag_extreme(var_col, threshold_col, op):
        value = l[var_col]
        threshold = l[threshold_col]
        flags = value > threshold if op == 'gt' else value < threshold
        unknown = value.isna() | threshold.isna()
        return flags.astype("boolean").mask(unknown, pd.NA)

    # Local labels (relative to day of year)
    l['extreme_tmax']     = flag_extreme('tmax',     'tmax_threshold',     'gt')
    l['extreme_tmin']     = flag_extreme('tmin',     'tmin_threshold',     'lt')
    l['extreme_prec']     = flag_extreme('prec',     'prec_threshold',     'gt')
    l['extreme_velmedia'] = flag_extreme('velmedia', 'velmedia_threshold', 'gt')
    l['extreme_racha']    = flag_extreme('racha',    'racha_threshold',    'gt')


    if MODE == 'global':
        # Global labels
        l['extreme_tmax_global']     = l['tmax']     > l['global_tmax_threshold']
        l['extreme_tmin_global']     = l['tmin']     < l['global_tmin_threshold']
        l['extreme_prec_global']     = l['prec']     > l['global_prec_threshold']
        l['extreme_velmedia_global'] = l['velmedia'] > l['global_velmedia_threshold']
        l['extreme_racha_global']    = l['racha']    > l['global_racha_threshold']
    else:
        l.drop(columns=[
            'global_tmax_threshold',
            'global_tmin_threshold',
            'global_prec_threshold',
            'global_velmedia_threshold',
            'global_racha_threshold'
        ], inplace=True)

    l.drop(columns=['c_month', 'c_day'], inplace=True)

    return l