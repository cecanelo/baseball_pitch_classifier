import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from sklearn.pipeline import Pipeline
from train import build_pipeline, suggest_params, run_hpo, save_best_params, load_best_params, refit_final_model, compute_metrics, save_pipeline, save_metrics #type: ignore
from utils import load_data #type: ignore
import pytest
import optuna
import json


def test_build_pipeline_returns_pipeline():
    result = build_pipeline('random_forest', {})
    assert isinstance(result, Pipeline)


def test_build_pipeline_adds_scaler_for_knn():
    result = build_pipeline('knn', {})
    assert 'scaler' in result.named_steps


def test_build_pipeline_adds_scaler_for_mlp():
    result = build_pipeline('mlp', {})
    assert 'scaler' in result.named_steps


def test_build_pipeline_no_scaler_for_decision_tree():
    result = build_pipeline('decision_tree', {})
    assert 'scaler' not in result.named_steps


def test_build_pipeline_returns_pipeline_for_decision_tree():
    result = build_pipeline('decision_tree', {})
    assert isinstance(result, Pipeline)


def test_build_pipeline_no_scaler_for_random_forest():
    result = build_pipeline('random_forest', {})
    assert 'scaler' not in result.named_steps


def test_build_pipeline_raises_unknown_model():
    with pytest.raises(ValueError):
        build_pipeline('unknown_model', {})


def test_suggest_params_int():
    study = optuna.create_study()
    trial = study.ask()
    search_space = {'n_estimators': {'type': 'int', 'low': 50, 'high': 300}}
    result = suggest_params(trial, search_space)
    assert 'n_estimators' in result
    assert isinstance(result['n_estimators'], int)
    assert 50 <= result['n_estimators'] <= 300


def test_suggest_params_float():
    study = optuna.create_study()
    trial = study.ask()
    search_space = {'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True}}
    result = suggest_params(trial, search_space)
    assert 'learning_rate' in result
    assert isinstance(result['learning_rate'], float)
    assert 0.01 <= result['learning_rate'] <= 0.3


def test_suggest_params_categorical_strings():
    study = optuna.create_study()
    trial = study.ask()
    search_space = {'max_features': {'type': 'categorical', 'choices': ['sqrt', 'log2']}}
    result = suggest_params(trial, search_space)
    assert 'max_features' in result
    assert result['max_features'] in ['sqrt', 'log2']


def test_suggest_params_categorical_lists():
    study = optuna.create_study()
    trial = study.ask()
    search_space = {'hidden_layer_size': {'type': 'categorical', 'choices': [[64, 32], [128, 64]]}}
    result = suggest_params(trial, search_space)
    assert 'hidden_layer_size' in result
    assert isinstance(result['hidden_layer_size'], list)
    assert result['hidden_layer_size'] in [[64, 32], [128, 64]]


def test_run_hpo_kfold_returns_best_params():
    X_train, X_test, y_train, y_test = load_data('data/processed')

    hpo_config = {
        'n_trials': 2,
        'metrics': {'primary': 'f1_weighted'},
        'search_space': {
            'n_estimators': {'type': 'int', 'low': 50, 'high': 100}}
        }
    val_config = {
        'strategy': 'kfold',
        'n_folds': 2
        }
    
    result = run_hpo('random_forest', X_train, y_train, hpo_config, val_config)

    assert isinstance(result, dict)
    assert 'n_estimators' in result


def test_run_hpo_fixed_split_returns_best_params():
    X_train, X_test, y_train, y_test = load_data('data/processed')

    hpo_config = {
        'n_trials': 2,
        'metrics': {'primary': 'f1_weighted'},
        'search_space': {
            'n_estimators': {'type': 'int', 'low': 50, 'high': 100}}
        }
    val_config = {
        'strategy': 'fixed_split',
        'val_size': 0.2
        }
    
    result = run_hpo('random_forest', X_train, y_train, hpo_config, val_config)

    assert isinstance(result, dict)
    assert 'n_estimators' in result


def test_save_and_load_best_params(tmp_path):
    best_params = {'n_estimators': 150, 'max_depth': 10}
    save_best_params(best_params, 'random_forest', str(tmp_path))
    result = load_best_params('random_forest', str(tmp_path))
    assert result == best_params


def test_load_best_params_raises_if_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_best_params('random_forest', str(tmp_path))


def test_refit_final_model_returns_pipeline():
    X_train, X_test, y_train, y_test = load_data('data/processed')
    best_params = {'n_estimators': 50, 'max_depth': 5}
    result = refit_final_model('random_forest', best_params, X_train, y_train)

    assert isinstance(result, Pipeline)
    assert hasattr(result.named_steps['model'], 'classes_')


def test_compute_metrics_returns_correct_structure():
    X_train, X_test, y_train, y_test = load_data('data/processed')
    best_params = {'n_estimators': 50, 'max_depth': 5}
    pipeline = refit_final_model('random_forest', best_params, X_train, y_train)
    result = compute_metrics(pipeline, X_test, y_test)

    assert isinstance(result, dict)
    assert 'weighted avg' in result
    assert 'f1-score' in result['weighted avg']


def test_save_pipeline_creates_file(tmp_path):
    pipeline = build_pipeline('random_forest', {'n_estimators': 50})
    save_pipeline(pipeline, 'random_forest', str(tmp_path))
    
    assert os.path.exists(os.path.join(str(tmp_path), 'random_forest.pkl'))


def test_save_metrics_creates_file(tmp_path):
    metrics = {'weighted avg': {'f1-score': 0.91}}
    save_metrics(metrics, 'random_forest', str(tmp_path))
    path = os.path.join(str(tmp_path), 'random_forest.json')
    assert os.path.exists(path)
    with open(path) as f:
        result = json.load(f)
    
    assert result == metrics


