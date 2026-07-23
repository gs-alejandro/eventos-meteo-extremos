import os                                           # acceso a ficheros
import logging                                      # logs
from dataset_building.era5.config import START_YEAR, END_YEAR, VARIABLES_BY_STAT
from dataset_building.era5.builder import get_era5_data, move_non_zips, unzip_files, combine_var_files

# preparamos logging
logging.basicConfig(
    filename=f'ERA5_CLM_build_events_{START_YEAR}_{END_YEAR}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

if __name__ == '__main__':

    # definimos años de los que obtener datos
    years = list(range(START_YEAR, END_YEAR+1))
    months = [f"{m:02d}" for m in range(1, 13)]

    # comprobamos que existen los directorios para guardar datos
    output_dir = 'era5_grid_data/compressed'            # ficheros iniciales
    os.makedirs(output_dir, exist_ok=True) 
    extracted_output_dir = 'era5_grid_data/extracted'   # ficheros descomprimidos
    os.makedirs(extracted_output_dir, exist_ok=True)
    combined_output_dir = 'era5_grid_data/combined'     # ficheros combinados por variable
    os.makedirs(combined_output_dir, exist_ok=True)

    # definir tareas (peticiones a Copernicus Data Store) para recopilar datos
    tasks = []
    for stat, vars in VARIABLES_BY_STAT.items():
        for year in years:
            for month in months:
                # !!!!!! NOTA IMPORTANTE: SOLO ES .nc SI SOLO SE DESCARGA UNA VARIABLE
                # EN CASO CONTRARIO ES UN .zip
                fname = f'CLM_{year}_{month}_{stat}'
                if len(vars) == 1:
                    fname += '.nc'
                else:
                    fname += '.zip'
                fpath = os.path.join(output_dir, fname)
                task = {
                    'year': year,
                    'month': month,
                    'stat': stat,
                    'vars': vars,
                    'fpath': fpath
                }
                tasks.append(task)

    for task in tasks:
        get_era5_data(task)

    move_non_zips(output_dir, extracted_output_dir)

    unzip_files(output_dir, extracted_output_dir)

    combine_var_files(extracted_output_dir, combined_output_dir)
