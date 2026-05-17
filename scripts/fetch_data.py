import pandas as pd
from time import sleep
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import pybaseball
import os
import argparse
from pybaseball import statcast
from utils import load_config

pybaseball.cache.enable()

# Statcast API is occasionally unreliable. Retry up to 3 times with a 5 second wait between attempts.
MAX_RETRIES = 3
    
def fetch_pitcher_data(start: str, end: str) -> pd.DataFrame:
    for attempt in range(MAX_RETRIES):
        try:
            df = statcast(start, end)
            return df
        except Exception as e:
            logging.warning(f'Attempt {attempt+1} failed: {e}')
            if attempt < MAX_RETRIES - 1:
                sleep(5)
    logging.warning(f'All {MAX_RETRIES} attempts failed.')
    return None

def save_pitcher_data(df: pd.DataFrame, csv_path: str) -> None:
    """
    Save a DataFrame to CSV, creating parent directories if needed.

    Args:
        df: DataFrame to save.
        csv_path: Full path including filename where the CSV will be written.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = load_config(args.config)

    start = config['season']['start']
    end = config['season']['end']
    raw_dir = config['paths']['raw_dir']

    season = start[:4]
    csv_path = f'{raw_dir}/pitch_data_{season}.csv'
    
    if os.path.exists(csv_path):
        logging.info('Dataset already exists.')
        response = input('Overwrite? (y/n): ')
        if response.lower() != 'y':
            logging.info('Aborting download')
            return
    
    df = fetch_pitcher_data(start, end)

    if df is None:
        logging.warning('Download failed.')
        return

    logging.info(f'Download successful. {len(df)} rows.')
    
    save_pitcher_data(df, csv_path)


if __name__ == "__main__":
    main()

            