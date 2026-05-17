import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

import pandas as pd
import os
import json
import argparse
import joblib
from sklearn.metrics import f1_score
from utils import load_config, load_data

def load_models(models_dir: str, model_names: list) -> dict:
    models = {}
    if not model_names:
        all_files = os.listdir(models_dir)
        for file in all_files:
            if file.endswith('.pkl'):
                model = file.replace('.pkl', '')
                model_names.append(model)
    
    for model in model_names:
        path = os.path.join(models_dir, f'{model}.pkl')
        if not os.path.exists(path):
            logging.warning(f'Model file not found: {path}. Skipping.')
            continue
        models[model] = joblib.load(path)
        logging.info(f'Loaded model: {model}')

    return models
    

def generate_predictions(models: dict, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    df = X_test.copy()
    df['y_true'] = y_test.values

    class_map = {i: label for i, label in enumerate(sorted(y_test.unique()))}

    for model_name, pipeline in models.items():
        predictions = pipeline.predict(X_test)
        if pd.api.types.is_numeric_dtype(pd.Series(predictions)):
            predictions = pd.Series(predictions).map(class_map).values
        df[f'{model_name}_pred'] = predictions

        probabilities = pipeline.predict_proba(X_test)
        class_labels = pipeline.named_steps['model'].classes_

        probabilities_strings = []
        for i in range(len(probabilities)):
            row_dict = dict(zip([str(c) for c in class_labels], [float(p) for p in probabilities[i]]))
            probabilities_strings.append(json.dumps(row_dict))
        
        df[f'{model_name}_probs'] = probabilities_strings
    
    return df
        

def save_predictions(df: pd.DataFrame, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info(f'Predictions saved to: {output_path}')


def print_summary(df: pd.DataFrame, model_names: list) -> None:
    logging.info('=== Evaluation Summary ===')

    for model_name in model_names:
        score = f1_score(df['y_true'], df[f'{model_name}_pred'], average='weighted')
        logging.info(f'{model_name}: weighted F1: {score:.4f}')
    
    correct = pd.DataFrame({m: df[f'{m}_pred'] == df['y_true'] for m in model_names})
    pct_all_correct = correct.all(axis=1).mean() * 100
    logging.info(f'Rows where all models correct: {pct_all_correct:.1f}%')

    first_model = model_names[0]
    agreed = pd.DataFrame({m: df[f'{m}_pred'] == df[f'{first_model}_pred'] for m in model_names})
    pct_agreed = agreed.all(axis=1).mean() * 100
    logging.info(f'Rows where all models agreed: {pct_agreed:.1f}%')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--models', nargs='*', default=[])
    args = parser.parse_args()

    config = load_config(args.config)

    processed_dir = config['paths']['processed_dir']
    models_dir    = config['paths']['models_dir'] if 'models_dir' in config['paths'] else 'results/models'
    metrics_dir   = config['paths']['metrics_dir'] if 'metrics_dir' in config['paths'] else 'results/metrics'

    X_train, X_test, y_train, y_test = load_data(processed_dir)

    models = load_models(models_dir, args.models)

    if not models:
        logging.warning('No models loaded. Aborting.')
        return

    df = generate_predictions(models, X_test, y_test)

    output_path = os.path.join(metrics_dir, 'predictions.csv')
    save_predictions(df, output_path)

    print_summary(df, list(models.keys()))


if __name__ == '__main__':
    main()
