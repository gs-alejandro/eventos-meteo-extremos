START_YEAR = 2010
END_YEAR = 2024

# definimos las variables agrupadas por tipo de estadística que necesitamos
VARIABLES_BY_STAT = {
    'daily_mean': [
        '10m_u_component_of_wind',
        '10m_v_component_of_wind',
        '2m_temperature',
        '2m_dewpoint_temperature',
        'mean_sea_level_pressure'
    ],
    'daily_max': [
        'maximum_2m_temperature_since_previous_post_processing',
        '10m_wind_gust_since_previous_post_processing',
        'mean_sea_level_pressure'
    ],
    'daily_min': [
        'minimum_2m_temperature_since_previous_post_processing',
        'mean_sea_level_pressure'
    ],
    'daily_sum': [
        'total_precipitation'
    ]
}
