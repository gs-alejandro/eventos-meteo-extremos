import pandas as pd 
import numpy as np  
import os
from tqdm import tqdm
import requests     # requests from API
import time         # waiting between requests
import logging      # keeping track of correct functioning
from datetime import datetime, timedelta # handling dates
from dataset_building.aemet.config import START_DATE, END_DATE, MAX_ATTEMPTS, BASE_URL, ADDED_URL, API_KEY, COLUMNS

def make_request(url, params=None, headers=None, max_retries=100, delay=20):
    '''
    Make and send the same request while allowing for retries

    Args: 
        url (string): URL where the request is sent
        params (dict): dictionary of request parameters
        headers (dict): dictionary of request headers
        max_retries (int): times a request will be attempted
        delay (int): delay after Too Many Requests

    Returns:
        If successful: response
        If failed: None
    '''
    retries = 0
    while retries < max_retries:
        # Short 1 second sleep as attempt to reduce cases of Too Many Requests
        time.sleep(1)
        # Try except to reattempt in cases of remote disconnection
        try: 
            response = requests.get(url, params=params, headers=headers, timeout=30)

            # OK
            if response.status_code == 200:
                return response
            
            # Too Many Requests
            elif response.status_code == 429:
                # print(f'Too Many Requests for {url}')
                # print(f'Retrying {retries+1} out of {max_retries} times after {delay} seconds')
                logging.warning(f'Too Many Requests for {url}, retrying {retries+1} out of {max_retries} times after {delay} seconds')
                time.sleep(delay)
                retries += 1

            # Other response
            else:
                # print(f'Failed for {url}')
                # print(f'Code: {response.status_code}')
                logging.warning(f'Failed for {url}, status code: {response.status_code}')
                logging.debug(f'Response detailed info: {response.text}')
                break

        except Exception as e:
            # print(f'Exception {e} for {url}')
            # print(f'Retrying {retries+1} out of {max_retries} after {delay} seconds')
            logging.warning(f'Exception {e} for {url}, retrying {retries+1} out of {max_retries} times after {delay} seconds')
            time.sleep(delay)
            retries += 1

    # Max retries reached
    return None


def build_date_strings(start, end):
    '''
    Change datetime objects into AEMET accepted scripts
    '''
    start_string = start.strftime('%Y-%m-%d') + 'T00:00:00UTC'
    end_string = end.strftime('%Y-%m-%d') + 'T00:00:00UTC'
    return start_string, end_string


def build_url(parameter_dict=None, base_url=BASE_URL, added_url=ADDED_URL):
    '''
    Function to build a URL from a dictionary, where every key is added with its value (if non falsy)
    '''
    if parameter_dict is None:
        parameter_dict = {}

    url = base_url + added_url

    for key, value in parameter_dict.items(): # .keys() is not needed but I include it for clarity
        url += f'/{key}'
        # Check if non falsy value, if so add to url
        if value:
            url += f'/{value}'

    return url


def fetch_aemet_data(start_date=START_DATE, end_date=END_DATE, output_file=None):
    '''
    Function to retrieve data from AEMET OpenData API
    '''

    if output_file is None:
        raise ValueError("build_aemet_data() missing required parameter: 'output_file'")

    # weather_data will exist if files have already been saved
    weather_data = None
    if os.path.exists(output_file):
        weather_data = pd.read_csv(output_file, usecols=['indicativo', 'fecha'])
    
    # flag to know if column names should be written in the output file
    first = False
    if weather_data is None:
        first = True        

    # Sliding window over days until end date is reached or surpassed (max window size allowed = 15)
    start = start_date
    delta = timedelta(days=14) # start day + 14 days = 15 day window
    total_chunks = (end_date - start_date).days // 15 + 1
    # while start <= end_date:
    for _ in tqdm(range(total_chunks), desc='Fetching AEMET data'):

        if start > end_date:
            break

        # Calculate new end (+14 days)
        end = start + delta 
        if end > end_date: # Edge case when surpassing final date
            end = end_date

        # If window already in df, skip
        if weather_data is not None and start.strftime('%Y-%m-%d') in weather_data['fecha'].values:
            logging.info(f'Data from {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')} already exists, moving to next chunk of data')
            # New window start = end + 1
            start = end + timedelta(days=1)
            continue

        start_string, end_string = build_date_strings(start, end)
        
        url_dict = {
            'fechaini': start_string,
            'fechafin': end_string,
            'todasestaciones': False
        }

        url = build_url(url_dict)
        params = {'api_key': API_KEY}
        headers = {'cache-control': 'no-cache'} # no cached info
        # print(url)
        logging.info(f'Attempting to fetch data from {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}')

        response = make_request(url=url, params=params, headers=headers)
        data_response = None
        attempts = 0

        # Attempt to fetch data from between same dates again if unsuccessful
        # Done because sometimes error is in response, but code is 200
        while not data_response:

            if attempts >= MAX_ATTEMPTS:
                logging.error(f"Couldn't add data from {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')} after {MAX_ATTEMPTS}, moving to next chunk of data")
                break 
            elif attempts > 0:
                logging.info(f'Reattempting to fetch data from {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}, attempt {attempts+1}')

            if response:
                # The actual data does not come within the response itself, it's in a link provided by the response
                json_data = response.json()
                # Response contains URLs to data and metadata, though only data will be used for now
                data_url = json_data.get('datos')

                if data_url:
                    # Send request data URL
                    data_response = make_request(url=data_url, params=params)
                    # Append to final data if recieved correctly

                    # If we have response, append to .csv file
                    if data_response:
                        data = data_response.json()
                        df_chunk = pd.DataFrame(data).reindex(columns=COLUMNS) # Make sure order is correct
                        # No column headers + no index
                        df_chunk.to_csv(output_file, mode='a', header=first, index=False)
                        first=False
                        logging.info(f'Successfully added data from {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}')

                else:
                    logging.warning(f'No data fetched from {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}, retrying')

            attempts+=1  

        # New window start = end + 1
        start = end + timedelta(days=1)

    return


def get_station_info():
    '''
    Function to get info about weather stations
    '''

    station_data = []

    url = build_url(added_url='/api/valores/climatologicos/inventarioestaciones/todasestaciones')
    params = {'api_key': API_KEY}
    headers = {'cache-control': 'no-cache'} # No cached info

    # print(url)

    logging.info(f'Attempting to fetch data of weather stations')

    # Make the request to the API
    response = make_request(url=url, params=params, headers=headers)
    
    data_response = None
    attempts = 0

    # Attempt to fetch data again if unsuccessful     
    while not data_response:

        if attempts >= MAX_ATTEMPTS:
            logging.error("Couldn't add data.")
            break 
        elif attempts > 0:
            logging.info(f'Reattempting to fetch data, attempt {attempts+1}')

        # If we got a response
        if response:
            # The actual data does not come within the response itself, it's in a link provided by the response
            json_data = response.json()

            # We get urls to data and metadata, though only data will be used for now
            data_url = json_data.get('datos')

            # Confirmamos que hemos conseguido datos
            if data_url:
                # We make a request to the data url
                data_response = make_request(url=data_url, params=params)

                # Append to final data if recieved correctly
                if data_response:
                    data = data_response.json()
                    for row in data:
                        station_data.append(dict(row))

                    logging.info(f'Successfully added data')
            
            else:
                logging.warning(f'No data fetched, retrying')

        attempts+=1  

    return station_data

