import pandas as pd
from time import sleep
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

import os
import argparse
from pybaseball import playerid_lookup, statcast_pitcher
from utils import load_config

# Statcast API is occasionally unreliable. Retry up to 3 times with a 5 second wait between attempts.
MAX_RETRIES = 3
    

def lookup_pitcher_id(first: str, last: str) -> int:
    """
    Look up a pitcher's MLBAM ID by name using the pybaseball playerid_lookup table.

    Args:
        first: Pitcher's first name.
        last: Pitcher's last name.

    Returns:
        MLBAM player ID as an integer.

    Raises:
        ValueError: If no player is found for the given name.
    """
    lookup_table = playerid_lookup(last, first)
    if lookup_table.empty:
        raise ValueError(f"No player found for {first} {last}. Check spelling and accents.")
    player_id = lookup_table['key_mlbam'].iloc[0]
    return player_id

def fetch_pitcher_data(pitcher_id: int, start: str, end: str) -> pd.DataFrame:
    """
    Download Statcast pitch-by-pitch data for one pitcher over a date range.
    Retries up to MAX_RETRIES times on failure.

    Args:
        pitcher_id: MLBAM player ID.
        start: Start date string in 'YYYY-MM-DD' format.
        end: End date string in 'YYYY-MM-DD' format.

    Returns:
        DataFrame of all pitches, or None if all retry attempts fail.
    """
    for attempt in range(MAX_RETRIES):
        try:
            pitcher_df = statcast_pitcher(start, end, pitcher_id)
            return pitcher_df
        except Exception as e:
            logging.warning(f'Attempt {attempt+1} failed: {e}')
            if attempt < MAX_RETRIES - 1:
                sleep(5)
    logging.warning(f"All {MAX_RETRIES} attempts failed for pitcher_id {pitcher_id}")
    return None

def get_missing_pitchers(pitcher_names: list, csv_path: str) -> list:
    """
    Return pitchers from pitcher_names that are not already in the saved CSV.

    Args:
        pitcher_names: List of dicts with 'first' and 'last' keys from data.yaml.
        csv_path: Path to the existing combined CSV file.

    Returns:
        Subset of pitcher_names whose data has not yet been downloaded.
    """
    if not os.path.exists(csv_path):
        return pitcher_names
    df = pd.read_csv(csv_path, usecols=['player_name'])
    existing_names = df['player_name'].unique()
    missing = []
    for p in pitcher_names:
        if f"{p['last']}, {p['first']}" not in existing_names:
            missing.append(p)
    return missing

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
    pitchers = config['pitchers']
    raw_dir = config['paths']['raw_dir']

    season = start[:4]
    csv_path = f'{raw_dir}/pitch_data_{season}.csv'

    missing_pitchers = get_missing_pitchers(pitchers, csv_path)

    if not missing_pitchers:
        logging.info('No pitchers missing.')
        return
    
    new_dataframes = []

    for pitcher in missing_pitchers:
        pitcher_id = lookup_pitcher_id(pitcher['first'], pitcher['last'])
        df = fetch_pitcher_data(pitcher_id, start, end)
        if df is not None:
            new_dataframes.append(df)

    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        new_dataframes.insert(0, existing_df)

    combined = pd.concat(new_dataframes, ignore_index=True)
    save_pitcher_data(combined, csv_path)


if __name__ == "__main__":
    main()

            