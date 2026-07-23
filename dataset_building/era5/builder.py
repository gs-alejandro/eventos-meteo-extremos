import os               # acceso a ficheros
import zipfile          # compresión
import cdsapi           # acceso a copernicus data store
import logging          # logs
import calendar         # sacar dias de cada mes
import shutil           # copiar archivos
import re               # expresiones regulares
import glob             # buscar archivos
from tqdm import tqdm   # barra de progreso
import xarray as xr     
from collections import defaultdict


# funcion que obtiene los datos de ERA5 dado una tarea
# entrada: tarea como definida arriba
# salida: guarda en fichero definido en tarea el archivo.nc/archivo.zip conteniendo los datos
def get_era5_data(task):
    # Si el fichero ya existe, continua
    if os.path.exists(task['fpath']):
        return
    
    # Obtener dias dinamicamente
    month_int = int(task['month'])
    num_days = calendar.monthrange(task['year'], month_int)[1]
    days = [f'{d:02d}' for d in range(1, num_days + 1)]

    # definimos cliente, dataset y request
    client = cdsapi.Client()
    dataset = 'derived-era5-single-levels-daily-statistics'
    request = {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': task['vars'],
        'year': str(task['year']),
        'month': [task['month']],
        'day': days,
        'daily_statistic': task['stat'],
        'time_zone': 'utc+00.00',
        'frequency': '3_hourly',
        # DATOS SACADOS DE https://es.wikipedia.org/wiki/Anexo:Puntos_extremos_de_Castilla-La_Mancha
        'area': [41.50, -5.75, 37.75, 1]  # [N, W, S, E] aproximado para encajonar CLM,
    }

    # intentamos obtener datos
    try:
         client.retrieve(dataset, request).download(task['fpath'])
    except Exception as e:
         print(f'ERROR: {e}')
         logging.error(f'Exception {e} for {task}')


# funcion que mueve los archivos .nc y los renombra
def move_non_zips(output_dir, extracted_output_dir):
    for filename in tqdm(os.listdir(output_dir), desc='Moving non-zip files'):
        if filename.endswith('_daily_sum.nc'):
            filepath = os.path.join(output_dir, filename)
            new_filename = filename.replace('_daily_sum.nc', '_daily_sum_precipitation.nc')
            new_filepath = os.path.join(extracted_output_dir, new_filename)
            shutil.copy2(filepath, new_filepath)
            # print(f'Copied: {filepath} to {new_filepath}')


# funcion que toma los archivos .zip en un directorio y extrae sus contenidos en otro
def unzip_files(output_dir, extracted_output_dir):
    # descomprime datos
    for filename in tqdm(os.listdir(output_dir), desc='Extracting zip files'):
        if filename.endswith(".zip"):
            zip_path = os.path.join(output_dir, filename)
            prefix = os.path.splitext(filename)[0]
            try:
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    for inner_file in zip_ref.namelist():
                        if inner_file.endswith(".nc"):
                            # Nombre: <prefijo>_<nombre_archivo_interno>
                            inner_name = os.path.basename(inner_file)
                            output_file = f"{prefix}_{inner_name}"
                            output_path = os.path.join(extracted_output_dir, output_file)
                            with zip_ref.open(inner_file) as source, open(output_path, "wb") as target:
                                target.write(source.read())
                            # print(f"Extracted and renamed: {output_file}")
            except Exception as e:
                print(f"Error in {filename}: {e}")


# funcion que toma los archivos .nc de distintas vars y meses y los combina por var
def combine_var_files(extracted_output_dir, combined_output_dir):
    # pattern = re.compile(r"_daily_([^_]+_.+?)\.nc$")
    pattern = re.compile(r'_daily_(.+?)\.nc$') # regex que captura entre _daily_ y .nc
    files = glob.glob(os.path.join(extracted_output_dir, '*.nc')) # archivos .nc del directorio 
  
    groups = defaultdict(list) # dict que crea valor por defecto al usar llave inexistente

    # agrupa archivos por variable
    for f in files: 
        filename = os.path.basename(f)
        match = pattern.search(filename)
        if match:
            key = match.group(1) # parte entre paréntesis de regex
            groups[key].append(f) 
            # print(key)

    # une datos de cada variable en eje temporal y los guarda en un archivo en directorio destino
    for key, group_files in tqdm(groups.items(), desc='Concatenating along time'):
        # print(f'Processing group {key} ({len(group_files)} files)')        
        ds_list = [xr.open_dataset(f) for f in group_files]
        time_dim = 'valid_time'
        combined = xr.concat(ds_list, dim=time_dim).sortby(time_dim)
        output_path = os.path.join(combined_output_dir, f'CLM_daily_{key}.nc')
        combined.to_netcdf(output_path)
        # print(f'Saved to {output_path}')

