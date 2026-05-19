# CLAUDE.md

## Project Overview

Baseball pitch classifier built as a learning exercise and portfolio project.
Uses 2023 MLB Statcast data to classify pitch types (FF, SL, CU, etc.) from
physical measurements using 6 machine learning models.

Target audience: ML practitioners and baseball fans via GitHub portfolio and LinkedIn.

UNDER NO CIRCUMSTANCE USE EM DASHES EVER IN THIS PROJECT!

---

## Pipeline

Run scripts in this order:

```
python scripts/fetch_data.py    --config configs/data.yaml
python scripts/preprocess.py    --config configs/data.yaml
python scripts/train.py         --config configs/data.yaml --model configs/model_dt.yaml
python scripts/train.py         --config configs/data.yaml --model configs/model_knn.yaml
python scripts/train.py         --config configs/data.yaml --model configs/model_rf.yaml
python scripts/train.py         --config configs/data.yaml --model configs/model_xgb.yaml
python scripts/train.py         --config configs/data.yaml --model configs/model_mlp.yaml
python scripts/train.py         --config configs/data.yaml --model configs/model_lr.yaml
python scripts/preprocess_specialist.py --config configs/data.yaml
python scripts/train.py         --config configs/data.yaml --model configs/model_xgb_specialist.yaml
python scripts/evaluate.py      --config configs/data.yaml --hierarchical
```

Analysis: `notebooks/result_analysis.ipynb`

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
  N_selection.ipynb          -- Determines n_per_class from July 2023 sample
  rule_based_classifier.ipynb -- Hand-crafted baseline with threshold derivation
  result_analysis.ipynb       -- Full model comparison and statistical analysis
results/
  models/             -- Saved model pipeline files (gitignored)
  metrics/            -- Per-model metrics JSON files and predictions CSV
  figures/            -- Confusion matrices, heatmaps, bootstrap distributions
scripts/
  fetch_data.py       -- Downloads all 2023 MLB Statcast data via pybaseball
  preprocess.py       -- Filters, samples, encodes, splits data
  train.py            -- Trains one model with Optuna HPO
  evaluate.py         -- Generates predictions CSV and evaluation summary
```

---

## Data

Source: MLB Statcast via pybaseball (`statcast()`)

Season: 2023 (2023-03-30 to 2023-10-01)

Scope: All MLB pitchers (720,684 raw pitches)

Raw file: `data/raw/pitch_data_2023.csv` (118 columns, ~720k rows)

Target variable: `pitch_type` (9 classes: FF, SI, FC, SL, ST, CU, KC, CH, FS)

Sampling: 10,000 rows per class globally (90,000 total, 72,000 train / 18,000 test)

---

## Features

11 features defined in `configs/data.yaml`:

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
| spin_axis | Spin axis angle (degrees) |

Full feature documentation: `docs/model_features.md`

---

## Key Design Decisions

**Raw data is never modified**
`fetch_data.py` saves all 118 Statcast columns. Filtering and feature selection
happen only in `preprocess.py`. This preserves the ability to add features
without re-downloading data.

**No pitcher identity leakage**
V1 trained on 8 pitchers with a random split, producing inflated F1 (0.98-0.99)
because the model learned pitcher signatures, not mechanics. V2 uses all MLB
pitchers with global sampling so the model must generalize across pitchers.

**Global sampling before train/test split**
10,000 rows per pitch type sampled globally after cleaning, before splitting.
This prevents class imbalance and ensures no pitcher dominates a class.

**Scaling is deferred to train.py, not preprocess.py**
Tree-based models (RF, XGBoost, DT) do not need scaling. KNN and MLP do.
Applying scaling per-model in `train.py` avoids applying it unnecessarily
and prevents test set leakage from a scaler fit on the full dataset.

**p_throws is encoded as 0/1, not kept as a string**
`release_pos_x` showed a bimodal distribution caused by pitcher handedness.
Adding `p_throws` gives the model context to interpret horizontal features
correctly. Encoded in `preprocess.py` before the train/test split.

**Stratified train/test split**
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

- SL (Slider) is the hardest class (mean F1=0.77). No single feature separates
  it from ST and FC. This is a data limitation, not a modeling artifact.
- 3.3% of test pitches are genuinely ambiguous: no model can classify them
  correctly and their feature values are indistinguishable from the rest of the test set.
- XGBoost label encoding is handled outside the pipeline via LabelEncoder in
  train.py. The encoder is not saved with the model. evaluate.py detects integer
  predictions and maps them back to string labels using sorted class order.
- `plate_x` and `plate_z` have ~1% outliers (wild pitches). Consider clipping
  in a future version of `preprocess.py`.

---

## Rule-Based Baseline

A hand-crafted if/else classifier was built as a baseline using mean feature
values derived from the training set. Thresholds are set as midpoints between
class means and refined via confusion matrix analysis.
Results: accuracy=0.54, weighted F1=0.5385.
Any trained model with F1 below 0.5385 should be investigated for bugs.
See `notebooks/rule_based_classifier.ipynb` for full analysis.

---

## Final Results

| Model | Weighted F1 | Training Time |
|---|---|---|
| Rule-based baseline | 0.5385 | N/A |
| Logistic Regression | 0.5991 | 05:21:52 |
| Decision Tree | 0.8206 | 00:02:59 |
| KNN | 0.8812 | 00:24:55 |
| Random Forest | 0.8910 | 01:43:51 |
| MLP | 0.8943 | 11:13:51 |
| XGBoost | 0.9081 | 01:16:01 |
| Hierarchical XGBoost | 0.9130 | 01:54:58 |

The hierarchical XGBoost is the best model (McNemar p=0.000002 vs base XGBoost).
Among single models, XGBoost is best. RF and MLP are statistically equivalent
(McNemar p=0.13). RF is the better practical choice given it trains in 1h 44m
vs MLP's 11h 13m.
