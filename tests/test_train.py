import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from sklearn.pipeline import Pipeline
from train import build_pipeline, suggest_params
import pytest
import optuna


def test_build_pipeline_returns_pipeline():
    result = build_pipeline('random_forest', {})
    assert isinstance(result, Pipeline)

def test_build_pipeline_adds_scaler_for_knn():
    result = build_pipeline('knn', {})
    assert 'scaler' in result.named_steps

def test_build_pipeline_adds_scaler_for_mlp():
    result = build_pipeline('mlp', {})
    assert 'scaler' in result.named_steps

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

'''
FUNCTION test_suggest_params_categorical_strings:
    CREATE an Optuna study
    CREATE a trial from the study

    SET search_space with one categorical parameter:
        name: "max_features", type: "categorical", choices: ["sqrt", "log2"]

    CALL suggest_params(trial, search_space)

    ASSERT "max_features" is in the result
    ASSERT the value is one of ["sqrt", "log2"]


FUNCTION test_suggest_params_categorical_lists:
    CREATE an Optuna study
    CREATE a trial from the study

    SET search_space with one categorical parameter:
        name: "hidden_layer_sizes", type: "categorical",
        choices: [[64, 32], [128, 64]]

    CALL suggest_params(trial, search_space)

    ASSERT "hidden_layer_sizes" is in the result
    ASSERT the value is a list, not a tuple
    ASSERT the value is one of [[64, 32], [128, 64]]
'''
            

