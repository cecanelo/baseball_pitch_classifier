import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from fetch_data import get_missing_pitchers


PITCHERS = [
    {'first': 'Shohei', 'last': 'Ohtani'},
    {'first': 'Spencer', 'last': 'Strider'},
    {'first': 'Gerrit', 'last': 'Cole'},
]


# ── get_missing_pitchers ──────────────────────────────────────────────────────

def test_get_missing_pitchers_returns_all_when_no_csv(tmp_path):
    csv_path = tmp_path / 'pitch_data_2023.csv'
    result = get_missing_pitchers(PITCHERS, str(csv_path))
    assert result == PITCHERS

def test_get_missing_pitchers_returns_only_missing(tmp_path):
    csv_path = tmp_path / 'pitch_data_2023.csv'
    df = pd.DataFrame({'player_name': ['Ohtani, Shohei', 'Cole, Gerrit']})
    df.to_csv(csv_path, index=False)

    result = get_missing_pitchers(PITCHERS, str(csv_path))
    assert len(result) == 1
    assert result[0]['last'] == 'Strider'

def test_get_missing_pitchers_returns_empty_when_all_present(tmp_path):
    csv_path = tmp_path / 'pitch_data_2023.csv'
    df = pd.DataFrame({
        'player_name': ['Ohtani, Shohei', 'Strider, Spencer', 'Cole, Gerrit']
    })
    df.to_csv(csv_path, index=False)

    result = get_missing_pitchers(PITCHERS, str(csv_path))
    assert result == []
