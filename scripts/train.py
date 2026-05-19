import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

import warnings
import time
import pandas as pd
import argparse
import os
import json
import joblib
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from utils import load_config, load_data

warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', message='Choices for a categorical distribution')



def build_pipeline(model_name: str, params: dict) -> Pipeline:
    model_map = {
        'logistic_regression': LogisticRegression,
        'decision_tree':       DecisionTreeClassifier, 
        'random_forest':       RandomForestClassifier,
        'xgboost':             XGBClassifier,
        'xgboost_specialist':  XGBClassifier,
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
    
    def log_trial(study, trial):
        ts = time.strftime('%H:%M:%S')
        is_best = ' — new best' if trial.value >= study.best_value else ''
        logging.info(f'[{ts}] Trial {trial.number + 1}/{hpo_config["n_trials"]} — '
                     f'score: {trial.value:.4f}{is_best}')

    logging.info(f'Starting HPO for {model_name} — {hpo_config["n_trials"]} trials')
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=hpo_config['n_trials'], callbacks=[log_trial])

    logging.info(f'Best score: {study.best_value:.4f}')
    for param, value in study.best_params.items():
        logging.info(f'  {param}: {value}')

    return study.best_params
       
def save_best_params(best_params: dict, model_name: str, metrics_dir: str) -> None:
    os.makedirs(metrics_dir, exist_ok=True)
    path = os.path.join(metrics_dir, f'{model_name}_best_params.json')
    with open(path, 'w') as f:
        json.dump(best_params, f, indent=2)
    logging.info(f'Best params saved to {path}')


def load_best_params(model_name: str, metrics_dir: str) -> dict:
    path = os.path.join(metrics_dir, f'{model_name}_best_params.json')
    if not os.path.exists(path):
        raise FileNotFoundError(f'Best params not found: {path}. Run with hpo.enabled=true first.')
    with open(path, 'r') as f:
        return json.load(f)


def refit_final_model(model_name: str, 
                      best_params: dict, 
                      X_train: pd.DataFrame, 
                      y_train: pd.Series) -> Pipeline:
    logging.info(f'Refitting final model with best HPO parameters for {model_name}')

    pipeline = build_pipeline(model_name, best_params)
    pipeline.fit(X_train, y_train)
    logging.info(f'Fitting of {model_name} complete.')

    return pipeline


def compute_metrics(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    logging.info('Computing metrics.')
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return report


def save_pipeline(pipeline: Pipeline, model_name: str, models_dir: str) -> None:
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, f'{model_name}.pkl')
    joblib.dump(pipeline, path)
    logging.info(f'Pipeline saved to {path}')


def save_metrics(metrics: dict, model_name: str, metrics_dir: str) -> None:
    os.makedirs(metrics_dir, exist_ok=True)
    path = os.path.join(metrics_dir, f'{model_name}.json')
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logging.info(f'Metrics saved to {path}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()

    data_config  = load_config(args.config)
    model_config = load_config(args.model)

    model_name    = model_config['model']['name']
    hpo_config    = model_config['hpo']
    val_config    = data_config['hpo_validation']
    processed_dir = data_config['paths']['processed_dir']
    models_dir    = model_config['paths']['models_dir']
    metrics_dir   = model_config['paths']['metrics_dir']

    train_x = model_config.get('data', {}).get('train_x', 'X_train.csv')
    train_y = model_config.get('data', {}).get('train_y', 'y_train.csv')
    X_train, X_test, y_train, y_test = load_data(processed_dir, train_x=train_x, train_y=train_y)

    start_time = time.perf_counter()

    if model_name in ['xgboost', 'xgboost_specialist']:
        le = LabelEncoder()
        y_train = pd.Series(le.fit_transform(y_train))
        mask = y_test.isin(le.classes_)
        X_test = X_test[mask]
        y_test = pd.Series(le.transform(y_test[mask]))
        logging.info(f'XGBoost: encoded labels {list(le.classes_)} → {list(range(len(le.classes_)))}')

    if hpo_config['enabled']:
        best_params = run_hpo(model_name, X_train, y_train, hpo_config, val_config)
        save_best_params(best_params, model_name, metrics_dir)
    else:
        best_params = load_best_params(model_name, metrics_dir)

    pipeline = refit_final_model(model_name, best_params, X_train, y_train)
    metrics  = compute_metrics(pipeline, X_test, y_test)
    logging.info(f'Weighted F1 ({model_name}): {metrics["weighted avg"]["f1-score"]:.4f}')

    elapsed = time.perf_counter() - start_time
    metrics['training_time_seconds'] = round(elapsed, 2)
    elapsed_fmt = time.strftime('%H:%M:%S', time.gmtime(elapsed))

    save_pipeline(pipeline, model_name, models_dir)
    save_metrics(metrics, model_name, metrics_dir)

    logging.info(f'Training complete for {model_name} in {elapsed_fmt}')


if __name__ == '__main__':
    main()


