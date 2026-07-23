import numpy as np
import pandas as pd
from datetime import datetime


# INFO FROM: 

# https://library.wmo.int/idurl/4/41650 (Vol 1 Chapter 3)
# World Meteorological Organization (WMO). Guide to Instruments and Methods of Observation (WMO-No. 8), Volume I. Geneva, 2024.
# https://library.wmo.int/viewer/68695/?offset=3#page=164&viewer=picture&o=bookmark&n=0&q=

# https://www.ngdc.noaa.gov/stp/space-weather/online-publications/miscellaneous/us-standard-atmosphere-1976/us-standard-atmosphere_st76-1562_noaa.pdf
# (Eq 33b, page 12)

# https://youtu.be/C8Its_imtyY

# Stull, R., 2017: "Practical Meteorology: An Algebra-based Survey of Atmospheric Science", Volume 9
# https://www.eoas.ubc.ca/books/Practical_Meteorology/


# Physical constants
g = 9.80665 # standard acceleration of gravity (m/s^2)
R = 287.05  # gas constant of dry air (J/kg·K)
a = 0.0065  # standard lapse rate (might not be real one, it's an estimation)
C_h = 0.12  # humidity coefficient


# INFO FROM: https://en.wikipedia.org/wiki/Clausius%E2%80%93Clapeyron_relation
# Applications > Meteorology and climatology 
def august_roche_magnus(t):
    '''
    Applies August-Roche-Magnus approximation of the Clausius-Clapeyron 
    equation for water vapor under typical conditions to estimate saturation vapor pressure

    Parameters:
        t: observed temperature (ºC)

    Returns:
        e_sat: saturation vapor pressure (hPa)
    '''
    e_sat = 6.1094 * np.exp((17.625 * t) / (t + 243.04))
    return e_sat


# https://www.ti.com/lit/pdf/snaa318
# https://preview.weather.gov/epz/wxcalc_vaporpressure
# |- https://www.weather.gov/media/epz/wxcalc/vaporPressure.pdf
def calculate_vapor_pressure(t, rh):
    '''
    Calculates vapor pressure using temperature and relative humidity

    Parameters:
        t: observed temperature (ºC)
        rh: relative humidity (%)

    Returns:
        e_s: vapor pressure
    '''
    # Calculate saturation vapor pressure
    e_sat = august_roche_magnus(t)
    # Calculate actual vapor pressure
    e_s = e_sat * (rh / 100)
    return e_s


# INFO FROM: https://library.wmo.int/idurl/4/41650 
# Vol I, Chapter 3, 3.7.2 General reduction formula (exponential form)
# https://www.eoas.ubc.ca/books/Practical_Meteorology/prmet102/Ch09-wxmaps-v102b.pdf
# Stull, R., 2017: "Practical Meteorology: An Algebra-based Survey of Atmospheric Science", Volume 9
def sea_level_pressure_reduction(p, t, h, rh, celsius=True):
    """
    Applies general sea level reduction formula as explained by WMO

    If pressure, temperature or altitude is NaN the NaN will propagate through
    the operation, resulting in a NaN value for that row/element in the returned series/array

    If relative humidity is NaN, a dry air column will be assumed, so humidity correction 
    (humid air is lighter than dry air) will be set to 0

    An important decision to take is to use this function with t and rh values:
        1. Mean daily temperature and mean relative humidity for all pressures
        2. Different values depending on pressure metric:
            a. Min temperature and max relative humidity for max pressure
            b. Max temperature and min relative humidity for min pressure
            c. Mean temperature and mean relative humidity for mean pressure

    Parameters:
        p : observed pressure series/array (hPa)
        t : temperature series/array (ºC/K)
        h: altitude of station series/array (m)
        rh: relative humidity series/array (%)
        celsius: flag of whether degrees are Celsius, if False, Kelvin will be assumed

    Returns:
        sl_p: pressure reduced to sea level series/array
    """


    t_C = t if celsius else t - 273.15
    t_K = t + 273.15 if celsius else t

    # Get vapor pressure and calculate correction
    e_s = calculate_vapor_pressure(t_C, rh)
    correction = (e_s * C_h)

    # When correction is missing, use 0 instead of NaN (assume dry air column)
    missing_correction = np.isnan(correction)
    correction = np.where(missing_correction, 0.0, correction)

    # Apply formula (exponential form)
    num = (g / R) * h
    den = t_K + (a * h)/2 + correction
    sl_p = p * np.exp(num / den)

    return sl_p


# https://qed.epa.gov/hms/meteorology/humidity/algorithms/
# https://bmcnoldy.earth.miami.edu/Humidity.html
def calculate_relative_humidity(td, t, celsius=True):
    '''
    Calculates relative humidity from dewpoint temperature and temperature

    Parameters:
        td: dewpoint temperature (K)
        t : temperature series/array (K)

    Returns:
        rh: relative humidity (%)
    '''
    t_c = t if celsius else t-273.15
    td_c = td if celsius else td-273.15

    num = np.exp(17.625*(td_c)/(td_c+243.04))
    den   = np.exp(17.625*(t_c)/(t_c+243.04))
    rh = 100 * num / den
    return rh


def build_missing_days_df(df, s, e):
    '''
    Builds a dataframe containing station id and date of missing registers
    '''
    full_ranges = []

    for uid, group in df.groupby('indicativo', observed=True):
        # Use 
        full_range = pd.date_range(start=datetime(s, 1, 1), end=datetime(e, 12, 31), freq='D')
        full_df = pd.DataFrame({'indicativo': uid, 'fecha': full_range})
        full_ranges.append(full_df)

    # Get dataframe of all date ranges
    full_df = pd.concat(full_ranges, ignore_index=True)

    # Merge dataframe
    merged = full_df.merge(df, on=['indicativo', 'fecha'], how='left', indicator=True)
    # Get 'left_only' rows, which are rows not present in dataframe
    missing = merged[merged['_merge'] == 'left_only']
    missing = missing[['indicativo', 'fecha']]
    missing

    return missing
