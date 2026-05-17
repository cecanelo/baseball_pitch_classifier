import pandas as pd
import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from fetch_data import fetch_pitcher_data, save_pitcher_data

def test_fetch_pitcher_data_returns_dataframe_on_success():
    dummy_df = pd.DataFrame({'pitch_type': ['FF', 'SL']})
    with patch('fetch_data.statcast', return_value=dummy_df):
        result = fetch_pitcher_data('2023-07-01', '2023-07-31')
    assert isinstance(result, pd.DataFrame)


def test_fetch_pitcher_data_returns_none_after_all_retries_fail():
    with patch('fetch_data.statcast', side_effect=Exception('API error')):
        result = fetch_pitcher_data('2023-07-01', '2023-07-31')
    assert result is None


def test_save_pitcher_data_creates_csv(tmp_path):
    df = pd.DataFrame({'pitch_type': ['FF', 'SL'], 'release_speed': [95.0, 88.0]})
    csv_path = tmp_path / 'pitch_data_2023.csv'
    save_pitcher_data(df, str(csv_path))
    assert csv_path.exists()
    assert pd.read_csv(csv_path).equals(df)
