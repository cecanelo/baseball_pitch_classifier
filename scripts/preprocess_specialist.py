import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

import pandas as pd
import argparse
import os
from utils import load_config
from preprocess import exclude_pitch_types, exclude_below_threshold, select_features, sample_per_class

ID_COLS = ['game_pk', 'at_bat_number', 'pitch_number']


def exclude_test_rows(df: pd.DataFrame, test_identifiers_path: str) -> pd.DataFrame:
    if not os.path.exists(test_identifiers_path):
        raise FileNotFoundError(f'Test identifiers not found: {test_identifiers_path}. Run preprocess.py first.')
    test_ids = pd.read_csv(test_identifiers_path)
    test_ids['_key'] = (test_ids['game_pk'].astype(str) + '_' +
                        test_ids['at_bat_number'].astype(str) + '_' +
                        test_ids['pitch_number'].astype(str))
    df['_key'] = (df['game_pk'].astype(str) + '_' +
                  df['at_bat_number'].astype(str) + '_' +
                  df['pitch_number'].astype(str))
    clean_df = df[~df['_key'].isin(test_ids['_key'])].drop(columns='_key')
    logging.info(f'Excluded {len(df) - len(clean_df)} test rows from pool')
    return clean_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = load_config(args.config)

    selected_features = config['features']
    target            = config['target']
    excluded_pitches  = config['filtering']['exclude_pitch_types']
    threshold         = config['filtering']['min_pitch_count']
    raw_dir           = config['paths']['raw_dir']
    processed_dir     = config['paths']['processed_dir']
    season            = config['season']['start'][:4]
    random_state      = config['split']['random_state']
    n_per_class       = config['specialist']['n_per_class']
    pitch_types       = config['specialist']['pitch_types']

    raw_csv_path = f'{raw_dir}/pitch_data_{season}.csv'
    logging.info(f'Loading raw data from {raw_csv_path}')

    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(f'Raw data not found: {raw_csv_path}. Run fetch_data.py first.')
    df = pd.read_csv(raw_csv_path)
    logging.info(f'Loaded {len(df)} rows')

    df = exclude_pitch_types(excluded_pitches, df)
    df = df[df['pitch_type'].isin(pitch_types)].copy()
    logging.info(f'After filtering to {pitch_types}: {len(df)} rows')

    df = exclude_below_threshold(df, threshold)

    df['p_throws'] = (df['p_throws'] == 'R').astype(int)
    df['pfx_x_x_p_throws'] = df['pfx_x'] * df['p_throws']

    test_identifiers_path = f'{processed_dir}/test_identifiers.csv'
    df = exclude_test_rows(df, test_identifiers_path)
    logging.info(f'After excluding test rows: {len(df)} rows')

    df = select_features(selected_features, target, df)
    df = df.dropna(subset=selected_features)
    logging.info(f'After dropping NaN rows: {len(df)} rows')

    df = sample_per_class(df, n_per_class, random_state)
    logging.info(f'After sampling: {len(df)} rows')
    logging.info(f'Class distribution:\n{df[target].value_counts().to_string()}')

    X_train = df[selected_features]
    y_train = df[target]

    os.makedirs(processed_dir, exist_ok=True)
    X_train.to_csv(f'{processed_dir}/X_train_specialist.csv', index=False)
    y_train.to_csv(f'{processed_dir}/y_train_specialist.csv', index=False)
    logging.info(f'Saved specialist training set to {processed_dir}')
    logging.info(f'Train: {len(X_train)} rows')


if __name__ == '__main__':
    main()
