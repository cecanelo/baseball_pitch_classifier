# CLAUDE.md

## Project Overview

Baseball pitch classifier built as a learning exercise and portfolio project.
Uses 2023 MLB Statcast data to classify pitch types (FF, SL, CU, etc.) from
physical measurements using 5 machine learning models.

Target audience: ML practitioners and baseball fans via GitHub portfolio and LinkedIn.

---

## Pipeline

Run scripts in this order:

```
python scripts/fetch_data.py    --config configs/data.yaml
python scripts/preprocess.py    --config configs/data.yaml
python scripts/train.py         --config configs/data.yaml --model configs/model_lr.yaml
python scripts/train.py         --config configs/data.yaml --model configs/model_rf.yaml
python scripts/train.py         --config configs/data.yaml --model configs/model_xgb.yaml
python scripts/train.py         --config configs/data.yaml --model configs/model_knn.yaml
python scripts/train.py         --config configs/data.yaml --model configs/model_mlp.yaml
python scripts/evaluate.py      --config configs/data.yaml
```

Final comparison and visualization: `notebooks/model_comparison.ipynb`

---

## Environment

Conda environment: `baseball_clf`

Activate: `conda activate baseball_clf`

Reproduce: `conda env create -f environment.yml`

---

## Project Structure

```
configs/              -- YAML configs for data pipeline and each model
data/
  raw/                -- Full Statcast CSV as downloaded (never modified)
  processed/          -- X_train, X_test, y_train, y_test CSVs
docs/                 -- Feature glossaries and analysis summaries
notebooks/
  data_exploration.ipynb    -- EDA, visualizations, rule-based baseline
  model_comparison.ipynb    -- Final model results and conclusions
results/
  models/             -- Saved model files
  metrics/            -- Per-model metrics JSON files
  figures/            -- Confusion matrices and feature importance plots
scripts/
  fetch_data.py       -- Downloads Statcast data via pybaseball
  preprocess.py       -- Filters, encodes, splits data
  train.py            -- Trains one model with Optuna HPO (not yet built)
  evaluate.py         -- Generates metrics and plots (not yet built)
```

---

## Data

Source: MLB Statcast via pybaseball (`statcast_pitcher`)

Season: 2023 (2023-03-30 to 2023-10-01)

Pitchers: 4 right-handed (Ohtani, Strider, Wheeler, Cole),
          4 left-handed (Kershaw, Snell, Fried, Valdez)

Raw file: `data/raw/pitch_data_2023.csv` (118 columns, ~20k rows)

Target variable: `pitch_type` (9 classes after filtering: FF, SI, FC, SL, ST, CU, KC, CH, FS)

---

## Features

10 features defined in `configs/data.yaml`:

| Feature | Description |
|---|---|
| release_speed | Pitch velocity at release (mph) |
| release_spin_rate | Spin rate at release (rpm) |
| pfx_x | Horizontal movement vs no-spin trajectory (ft) |
| pfx_z | Vertical movement vs no-spin trajectory (ft) |
| release_pos_x | Horizontal release position (ft) |
| release_pos_z | Vertical release position (ft) |
| plate_x | Horizontal plate location (ft) |
| plate_z | Vertical plate location (ft) |
| release_extension | Extension toward plate at release (ft) |
| p_throws | Pitcher handedness, encoded as 1=R / 0=L |

Full feature documentation: `docs/model_features.md`

---

## Key Design Decisions

**Raw data is never modified**
`fetch_data.py` saves all 118 Statcast columns. Filtering and feature selection
happen only in `preprocess.py`. This preserves the ability to add features
without re-downloading data.

**Scaling is deferred to train.py, not preprocess.py**
Tree-based models (RF, XGBoost) do not need scaling. KNN and MLP do.
Applying scaling per-model in `train.py` avoids applying it unnecessarily
and prevents test set leakage from a scaler fit on the full dataset.

**p_throws is encoded as 0/1, not kept as a string**
`release_pos_x` showed a bimodal distribution caused by pitcher handedness.
Adding `p_throws` gives the model context to interpret horizontal features
correctly. Encoded in `preprocess.py` before the train/test split.

**Stratified train/test split**
Class imbalance is significant (FF ~38% of pitches, FS ~0.6%).
`stratify=True` in `data.yaml` ensures class proportions are preserved
in both train and test sets.

**HPO via Optuna**
Each model config defines a search space. `train.py` runs 50 Optuna trials
per model optimizing weighted F1. The best parameters are used for the
final model saved to `results/models/`.

---

## Code Conventions

- All scripts accept `--config` as a CLI argument pointing to `configs/data.yaml`
- Functions take `df: pd.DataFrame` as the first parameter
- Type hints on all function signatures
- `load_config(config_path: str) -> dict` lives in `scripts/utils.py` and is
  imported by all scripts
- No hardcoded file paths -- all paths come from config
- Logging via the `logging` module, configured before library imports to prevent
  pybaseball from overriding the log level

---

## Known Limitations

- Dataset covers only 8 pitchers from the 2023 season. The classifier partially
  learns pitcher-specific signatures rather than universal pitch mechanics.
  Performance on unseen pitchers may degrade.
- Class imbalance is significant. FF has ~38% of pitches, FS has ~0.6%.
  Models are evaluated with weighted F1 to account for this.
- Rule-based baseline achieves accuracy=0.753, F1=0.735. ML models must beat this.
- CU/KC and FF/FC are the hardest pairs to separate. This is inherent to the data,
  not a modeling artifact.
- `plate_x` and `plate_z` have ~1% outliers (wild pitches). Consider clipping
  in a future version of `preprocess.py`.

---

## Rule-Based Baseline

A hand-crafted if/else classifier was built during EDA as a baseline.
Results: accuracy=0.753, weighted F1=0.735.
Any trained model with F1 below 0.735 should be investigated for bugs.
See `notebooks/data_exploration.ipynb` for full confusion matrix and analysis.
