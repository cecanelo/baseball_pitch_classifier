import yaml
import os
import pandas as pd
import logging


def load_config(config_path: str) -> dict:
    """
    Load and parse a YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Parsed config as a dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the file is not valid YAML.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse config file {config_path}: {e}")
        

def load_data(processed_dir: str, train_x: str = 'X_train.csv', train_y: str = 'y_train.csv') -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    for filename in [train_x, 'X_test.csv', train_y, 'y_test.csv']:
        path = os.path.join(processed_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f'Processed file not found: {path}. Run preprocess.py first.')

    X_train = pd.read_csv(os.path.join(processed_dir, train_x))
    X_test  = pd.read_csv(os.path.join(processed_dir, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(processed_dir, train_y)).squeeze()
    y_test  = pd.read_csv(os.path.join(processed_dir, 'y_test.csv')).squeeze()

    logging.info(f'Loaded {len(X_train)} training rows.')
    logging.info(f'Loaded {len(X_test)} test rows.')

    return X_train, X_test, y_train, y_test



