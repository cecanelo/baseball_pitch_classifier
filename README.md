# Baseball Pitch Classifier

A machine learning project that classifies MLB pitch types from physical measurements using 2023 Statcast data.

> **Work in progress.** This repository is under active development. Results and documentation will be updated as the project progresses.

## Overview

- **Goal:** Classify pitch types (FF, SL, CU, etc.) from physical measurements alone — no pitcher identity
- **Data:** 2023 MLB Statcast — all pitchers, 9 pitch types, 90,000 samples (10,000 per class)
- **Models:** Decision Tree, KNN, Random Forest, XGBoost
- **Best result:** XGBoost — weighted F1 = 0.9091 (global model, no pitcher leakage)

## Quickstart

```bash
conda env create -f environment.yml
conda activate baseball_clf

python scripts/fetch_data.py    --config configs/data.yaml
python scripts/preprocess.py    --config configs/data.yaml
python scripts/train.py         --config configs/data.yaml --model configs/model_xgb.yaml
```

## Project Structure

```
configs/       YAML configs for data pipeline and each model
data/          Raw and processed data
notebooks/     EDA, N selection analysis, model comparison
scripts/       fetch_data, preprocess, train, evaluate
tests/         Unit tests (35 passing)
results/       Saved models, metrics, figures
docs/          Feature documentation
```

## Key Design Decisions

- **No pitcher identity** in features — model generalizes to unseen pitchers
- **Global sampling** — 10,000 rows per pitch type to balance classes
- **HPO via Optuna** — 50 trials per model optimizing weighted F1
- **Stratified train/test split** — preserves class proportions

## Results (v2 — all MLB pitchers, 2023)

| Model | Weighted F1 | Training Time |
|---|---|---|
| Decision Tree | 0.8205 | 00:02:59 |
| KNN | 0.8815 | 00:24:55 |
| Random Forest | 0.8907 | 01:43:51 |
| XGBoost | **0.9091** | 01:16:01 |

Rule-based baseline: 0.735
