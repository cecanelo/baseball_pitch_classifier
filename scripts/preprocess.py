import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

import pandas as pd
import argparse
import os
from sklearn.model_selection import train_test_split
from utils import load_config


def select_features(selected_features: list, target: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the model features and target column, dropping everything else.

    Args:
        selected_features: List of feature column names from data.yaml.
        target: Name of the target column (pitch_type).
        df: Filtered Statcast DataFrame.

    Returns:
        DataFrame containing only the selected features and target column.
    """
    cols_to_keep = selected_features + [target]
    clean_df = df[cols_to_keep]
    return clean_df

def exclude_pitch_types(excluded_pitches: list, df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with null pitch_type and rows matching excluded pitch type codes.

    Args:
        excluded_pitches: List of pitch type codes to remove (e.g. ['PO', 'UN']).
        df: Raw Statcast DataFrame.

    Returns:
        DataFrame with excluded and null pitch types removed.
    """
    df = df.dropna(subset=['pitch_type'])
    clean_df = df[~df['pitch_type'].isin(excluded_pitches)]
    return clean_df

def exclude_below_threshold(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """
    Remove rows belonging to pitch types with fewer than threshold samples.

    Args:
        df: Statcast DataFrame with pitch_type column.
        threshold: Minimum number of samples required to keep a pitch type.

    Returns:
        DataFrame with rare pitch types removed.
    """
    pitch_counts = df['pitch_type'].value_counts()
    rare_pitches = pitch_counts[pitch_counts < threshold].index
    clean_df = df[~df['pitch_type'].isin(rare_pitches)]
    return clean_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = load_config(args.config)

    selected_features = config['features']
    target = config['target']
    excluded_pitches = config['filtering']['exclude_pitch_types']
    threshold = config['filtering']['min_pitch_count']
    raw_dir = config['paths']['raw_dir']
    processed_dir = config['paths']['processed_dir']
    season = config['season']['start'][:4]
    test_size = config['split']['test_size']
    random_state = config['split']['random_state']
      
    raw_csv_path = f'{raw_dir}/pitch_data_{season}.csv'
    logging.info(f'Loading raw data from {raw_csv_path}')
    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(f"Raw data not found: {raw_csv_path}. Run fetch_data.py first.")
    df = pd.read_csv(raw_csv_path)
    logging.info(f'Loaded {len(df)} rows')

    df = exclude_pitch_types(excluded_pitches, df)
    logging.info(f'After excluding pitch types: {len(df)} rows')

    df = exclude_below_threshold(df, threshold)
    logging.info(f'After threshold filter: {len(df)} rows, classes: {sorted(df["pitch_type"].unique())}')

    # release_pos_x is bimodal due to handedness. Encoding p_throws as 0/1 gives
    # the model context to interpret horizontal features correctly.
    df['p_throws'] = (df['p_throws'] == 'R').astype(int)
    df = select_features(selected_features, target, df)

    X = df[selected_features]
    y = df[target]
    stratify = y if config['split']['stratify'] else None

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size, stratify=stratify,
                                                        random_state=random_state)

    os.makedirs(processed_dir, exist_ok=True)
    X_train.to_csv(f'{processed_dir}/X_train.csv', index=False)
    X_test.to_csv(f'{processed_dir}/X_test.csv', index=False)
    y_train.to_csv(f'{processed_dir}/y_train.csv', index=False)
    y_test.to_csv(f'{processed_dir}/y_test.csv', index=False)
    logging.info(f'Saved train/test splits to {processed_dir}')
    logging.info(f'Train: {len(X_train)} rows, Test: {len(X_test)} rows')


if __name__ == "__main__":
    main()
