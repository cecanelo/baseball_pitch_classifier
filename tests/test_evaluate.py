import pandas as pd
import numpy as np
import json
import sys
import os
import joblib
from unittest.mock import MagicMock
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from evaluate import load_models, generate_predictions, save_predictions, generate_hierarchical_predictions


def make_real_pipeline():
    return Pipeline([('model', DecisionTreeClassifier())])


# ── load_models ───────────────────────────────────────────────────────────────

def test_load_models_loads_requested_models(tmp_path):
    joblib.dump(make_real_pipeline(), tmp_path / 'xgboost.pkl')
    result = load_models(str(tmp_path), ['xgboost'])
    assert 'xgboost' in result

def test_load_models_skips_missing_models(tmp_path):
    result = load_models(str(tmp_path), ['nonexistent_model'])
    assert result == {}

def test_load_models_loads_all_when_no_names_given(tmp_path):
    joblib.dump(make_real_pipeline(), tmp_path / 'random_forest.pkl')
    joblib.dump(make_real_pipeline(), tmp_path / 'xgboost.pkl')

    result = load_models(str(tmp_path), [])
    assert 'random_forest' in result
    assert 'xgboost' in result


# ── generate_predictions ──────────────────────────────────────────────────────

def test_generate_predictions_has_correct_columns():
    X_test = pd.DataFrame({'release_speed': [95.0, 88.0], 'pfx_z': [1.2, 0.3]})
    y_test = pd.Series(['FF', 'SL'])

    pipeline = MagicMock()
    pipeline.predict.return_value = np.array(['FF', 'SL'])
    pipeline.predict_proba.return_value = np.array([[np.float32(0.9), np.float32(0.1)],
                                                    [np.float32(0.2), np.float32(0.8)]])
    pipeline.named_steps = {'model': MagicMock(classes_=np.array(['FF', 'SL']))}

    models = {'xgboost': pipeline}
    result = generate_predictions(models, X_test, y_test)

    assert 'y_true' in result.columns
    assert 'xgboost_pred' in result.columns
    assert 'xgboost_probs' in result.columns

def test_generate_predictions_correct_row_count():
    X_test = pd.DataFrame({'release_speed': [95.0, 88.0, 82.0]})
    y_test = pd.Series(['FF', 'SL', 'CU'])

    pipeline = MagicMock()
    pipeline.predict.return_value = np.array(['FF', 'SL', 'CU'])
    pipeline.predict_proba.return_value = np.array([[np.float32(0.9), np.float32(0.05), np.float32(0.05)]] * 3)
    pipeline.named_steps = {'model': MagicMock(classes_=np.array(['FF', 'SL', 'CU']))}

    result = generate_predictions({'xgboost': pipeline}, X_test, y_test)
    assert len(result) == 3

def test_generate_predictions_probs_are_valid_json():
    X_test = pd.DataFrame({'release_speed': [95.0]})
    y_test = pd.Series(['FF'])

    pipeline = MagicMock()
    pipeline.predict.return_value = np.array(['FF'])
    pipeline.predict_proba.return_value = np.array([[np.float32(0.9), np.float32(0.1)]])
    pipeline.named_steps = {'model': MagicMock(classes_=np.array([np.int64(0), np.int64(1)]))}

    result = generate_predictions({'xgboost': pipeline}, X_test, y_test)
    parsed = json.loads(result['xgboost_probs'].iloc[0])
    assert '0' in parsed
    assert '1' in parsed


# ── generate_hierarchical_predictions ────────────────────────────────────────

def test_generate_hierarchical_predictions_returns_series():
    # sorted unique: FC=0, FF=1, SL=2, ST=3
    # predictions: FF, SL, ST, FC -> 1, 2, 3, 0
    # specialist_mask: False, True, True, True (SL, ST, FC)
    # specialist predicts 3 rows: FC=0, SL=1, ST=2
    X_test = pd.DataFrame({'release_speed': [95.0, 85.0, 82.0, 88.0]})
    y_test = pd.Series(['FF', 'SL', 'ST', 'FC'])

    xgb = MagicMock()
    xgb.predict.return_value = np.array([np.int64(1), np.int64(2), np.int64(3), np.int64(0)])

    specialist = MagicMock()
    specialist.predict.return_value = np.array([np.int64(1), np.int64(2), np.int64(0)])

    result = generate_hierarchical_predictions(xgb, specialist, X_test, y_test)
    assert isinstance(result, pd.Series)
    assert len(result) == 4


def test_generate_hierarchical_predictions_non_specialist_unchanged():
    X_test = pd.DataFrame({'release_speed': [95.0, 85.0]})
    y_test = pd.Series(['FF', 'SL', 'ST', 'FC', 'CH', 'CU', 'KC', 'SI', 'FS'])

    xgb = MagicMock()
    xgb.predict.return_value = np.array([np.int64(3), np.int64(3)])

    specialist = MagicMock()
    specialist.predict.return_value = np.array([], dtype=np.int64)

    result = generate_hierarchical_predictions(xgb, specialist, X_test, y_test)
    assert all(result == 'FF')


# ── save_predictions ──────────────────────────────────────────────────────────

def test_save_predictions_creates_file(tmp_path):
    df = pd.DataFrame({'y_true': ['FF', 'SL'], 'xgboost_pred': ['FF', 'CU']})
    output_path = str(tmp_path / 'metrics' / 'predictions.csv')
    save_predictions(df, output_path)
    assert os.path.exists(output_path)

def test_save_predictions_csv_contents(tmp_path):
    df = pd.DataFrame({'y_true': ['FF', 'SL'], 'xgboost_pred': ['FF', 'CU']})
    output_path = str(tmp_path / 'predictions.csv')
    save_predictions(df, output_path)
    result = pd.read_csv(output_path)
    assert list(result.columns) == ['y_true', 'xgboost_pred']
    assert len(result) == 2
