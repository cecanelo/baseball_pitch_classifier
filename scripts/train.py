import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

import pandas as pd
import argparse
import os
import json
import joblib
import optuna
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost import XGBClassifier
from utils import load_config, load_data


def build_pipeline(model_name: str, params: dict) -> Pipeline:
    model_map = {
        'logistic_regression': LogisticRegression,
        'random_forest':       RandomForestClassifier,
        'xgboost':             XGBClassifier,
        'knn':                 KNeighborsClassifier,
        'mlp':                 MLPClassifier,
        }

    if model_name not in model_map:
        raise ValueError(f"Unknown model name: '{model_name}'. Valid options: {list(model_map.keys())}")

    model = model_map[model_name](**params)

    if model_name in ['knn', 'mlp']:
        return Pipeline([("scaler", StandardScaler()), ("model", model)])
    else:
        return Pipeline([("model", model)])


def suggest_params(trial: optuna.Trial, search_space: dict) -> dict:
    params = {}
    for param_name, spec in search_space.items():
        if spec['type'] == 'int':
            params[param_name] = trial.suggest_int(param_name, spec['low'], spec['high'])
        elif spec['type'] == 'float':
            params[param_name] = trial.suggest_float(param_name, spec['low'], spec['high'], log=spec.get('log', False))
        elif spec['type'] == 'categorical':
            choices = []
            for c in spec['choices']:
                if isinstance(c, list):
                    choices.append(tuple(c))
                else:
                    choices.append(c)
            values = trial.suggest_categorical(param_name, choices)

            if isinstance(values, tuple):
                values = list(values)

            params[param_name] = values
    return params


def run_hpo(model_name: str, 
            X_train: pd.DataFrame, 
            y_train: pd.Series, 
            hpo_config: dict,
            val_config: dict
            ) -> dict:
    
    def objective(trial):
        sampled_params = suggest_params(trial, hpo_config['search_space'])
        pipeline = build_pipeline(model_name, sampled_params)

        if val_config['strategy'] == 'kfold':
            scores = cross_val_score(pipeline, 
                                     X_train, 
                                     y_train, 
                                     cv=val_config['n_folds'],
                                     scoring=hpo_config['metrics']['primary']
                                     )
            return scores.mean()

        elif val_config['strategy'] == 'fixed_split':
            X_tr, X_val, y_tr, y_val = train_test_split(X_train, 
                                                        y_train, 
                                                        test_size=val_config['val_size']
                                                        )
            pipeline.fit(X_tr, y_tr)
            y_pred = pipeline.predict(X_val)
            return f1_score(y_val, y_pred, average='weighted')
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=hpo_config['n_trials'])

    logging.info(f'Best score: {study.best_value:.4f}')
    for param, value in study.best_params.items():
        logging.info(f'  {param}: {value}')

    return study.best_params
       


'''FUNCTION run_hpo(model_name, X_train, y_train, hpo_config, val_config):

    DEFINE inner function objective(trial):

        CALL suggest_params(trial, hpo_config['search_space'])
            → gives us a dict of sampled hyperparameters

        CALL build_pipeline(model_name, sampled params)
            → gives us a fitted-ready Pipeline

        IF val_config['strategy'] is 'kfold':
            SCORE the pipeline using cross_val_score
                with cv = val_config['n_folds']
                and scoring = 'f1_weighted'
            RETURN the mean score across all folds

        ELSE IF val_config['strategy'] is 'fixed_split':
            SPLIT X_train into X_tr, X_val, y_tr, y_val
                using val_config['val_size']
            FIT the pipeline on X_tr, y_tr
            SCORE on X_val, y_val using f1_weighted
            RETURN that score

    CREATE an Optuna study with direction = 'maximize'
        (we want to maximize F1, not minimize)

    RUN study.optimize(objective, n_trials = hpo_config['n_trials'])

    LOG the best score and best parameters found

    RETURN study.best_params
'''
