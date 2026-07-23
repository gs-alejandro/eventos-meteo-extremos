STATION_SELECTION_PARAMS =  {
    'missing_days': 0.01,
    'min_date': '2010-01-01',
    'min_coverage': '2024-01-01',
    'nan_threshold': 0.20,
    'threshold_cols': [
        'tmed',
        'tmax',
        'tmin',
        'velmedia',
        'racha',
        # 'dir',
        'hrMedia',
        'prec',
        'presMax',
        'presMin',
        # 'sol'
    ],
    'max_altitude': 750,
}

LOWER_LIMIT_YEAR_AEMET = 1990

PROVINCES_CLM = ['ALBACETE', 'CUENCA', 'GUADALAJARA', 'TOLEDO', 'CIUDAD REAL']

COLS_TO_DROP = [
    'precIp',
    'precAcum',
    'horaPresMax',
    'horaPresMin',
    'horaPresMaxVarias',
    'horaPresMinVarias',
    'hrMax',
    'hrMin',
    'horaHrMax',
    'horaHrMin',
    'horaHrMaxVarias',
    'horaHrMinVarias',
    'horaracha',
    'horatmin',
    'horatmax',
    'horatmaxVarias',
    'horatminVarias',
    'sol', # QUIZÁ BORRAR
    'dir',
    'dirVarias'
]
