import pandas as pd
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from utils import load_data


def test_load_data_returns_correct_types():
    X_train, X_test, y_train, y_test = load_data('data/processed')

    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0

def test_load_data_raises_if_directory_is_missing():
    with pytest.raises(FileNotFoundError):
        load_data('data/nonexistent')


def test_load_data_loads_custom_train_files():
    X_train, X_test, y_train, y_test = load_data(
        'data/processed',
        train_x='X_train_specialist.csv',
        train_y='y_train_specialist.csv'
    )
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert len(X_train) > 0


def test_load_data_raises_if_custom_train_file_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data(str(tmp_path), train_x='nonexistent.csv', train_y='nonexistent.csv')