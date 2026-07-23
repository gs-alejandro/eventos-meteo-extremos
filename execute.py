import subprocess
import os
from datetime import datetime
from dataset_building.aemet.config import START_DATE, END_DATE

if __name__ == '__main__':
    subprocess.run(['python', '-m', 'scripts.build_aemet'], check=True)
    subprocess.run(['python', '-m', 'scripts.clean_aemet'], check=True)
    subprocess.run(['python', '-m', 'scripts.build_era5'], check=True)
    subprocess.run(['python', '-m', 'scripts.process_dataset'], check=True)
    subprocess.run(['python', '-m', 'scripts.label_dataset'], check=True)
