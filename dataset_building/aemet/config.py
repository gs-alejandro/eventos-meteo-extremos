import os
from dotenv import load_dotenv
from datetime import datetime

# Load .env and read AEMET OpenData Key
load_dotenv()
API_KEY = os.getenv("AEMET_API_KEY")

# DATES FOR DATA: YYYY-MM-DD (AAAA-MM-DD)
START_DATE = datetime(1970, 1, 1)
END_DATE = datetime(2024, 12, 31)

# 
MAX_ATTEMPTS = 50
BASE_URL = 'https://opendata.aemet.es/opendata' # Base URL from AEMET OpenData
ADDED_URL = '/api/valores/climatologicos/diarios/datos' # Rest of URL before parameters

COLUMNS = [
    # DATOS DE REGISTRO
    'fecha',        # Fecha de registro / Register date
    'indicativo',   # Identificador de estación meteorológica 
    'nombre',       # Nombre de estación meteorológica
    'provincia',    # Provincia
    'altitud',

    # DATOS DE TEMPERATURA
    'tmed',         # Temperatura media del día
    'tmax',         # Temperatura máxima del día
    'tmin',         # Temperatura mínima del día
    'horatmax',     # Hora de temperatura máxima
    'horatmin',     # Hora de temperatura mínima

    # DATOS DE VIENTO
    'velmedia',     # Velocidad media del viento
    'racha',        # Velocidad máxima de viento
    'dir',          # Dirección de mayor racha de viento
    'horaracha',    # Hora de mayor racha de viento

    # DATOS DE HUMEDAD
    'hrMedia',      # Humedad media
    'hrMax',        # Humedad máxima
    'hrMin',        # Humedad mínima
    'horaHrMax',    # Hora de humedad máxima
    'horaHrMin',    # Hora de humedad mínima

    # DATOS DE PRECIPITACIONES
    'prec',         # Cantidad de precipitación

    # DATOS DE INSOLACIÓN
    'sol',          # Horas de insolación

    # DATOS DE PRESIÓN
    'presMax',      # Presión máxima
    'presMin',      # Presión mínima
    'horaPresMax',  # Hora de presión máxima
    'horaPresMin',  # Hora de presión mínima
]
