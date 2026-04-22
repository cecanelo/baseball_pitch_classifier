import pandas as pd
import yaml
import argparse
import os
from sklearn.model_selection import train_test_split


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def select_features(selected_features: list, target: str, df: pd.DataFrame) -> pd.DataFrame:
    cols_to_keep = selected_features + [target]
    clean_df = df[cols_to_keep]
    return clean_df

def exclude_pitch_types(excluded_pitches: list, df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=['pitch_type'])
    clean_df = df[~df['pitch_type'].isin(excluded_pitches)]
    return clean_df

def exclude_below_threshold(threshold: int, df: pd.DataFrame) -> pd.DataFrame:
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
    # processed_csv_path = f'{processed_dir}/pitch_data_{season}.csv'
    df = pd.read_csv(raw_csv_path)

    df = exclude_pitch_types(excluded_pitches, df)
    df = exclude_below_threshold(threshold, df)
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

if __name__ == "__main__":
    main()
