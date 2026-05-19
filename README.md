# Baseball Pitch Classifier

A machine learning project that classifies MLB pitch types from physical
measurements using 2023 Statcast data.

## Motivation

The goal was to build a complete ML pipeline from data collection to evaluation,
using a domain that makes the results easy to reason about (at least to me).

## Results (2023 MLB, all pitchers, 9 pitch types)

| Model | Weighted F1 | Training Time |
|---|---|---|
| Rule-based baseline | 0.5385 | N/A |
| Logistic Regression | 0.5991 | 05:21:52 |
| Decision Tree | 0.8206 | 00:02:59 |
| KNN | 0.8812 | 00:24:55 |
| Random Forest | 0.8910 | 01:43:51 |
| MLP | 0.8943 | 11:13:51 |
| XGBoost | 0.9081 | 01:16:01 |
| Hierarchical XGBoost | **0.9130** | 01:54:58 |

The hierarchical XGBoost combines a 9-class base model with a specialist
trained on SL, FC, and ST only. Its improvement over base XGBoost is
statistically significant (McNemar p=0.000002). Random Forest and MLP are
statistically equivalent (McNemar p=0.13) despite MLP taking 6x longer to
train.

## Key Findings

- **SL (Slider) is the hardest pitch type** (mean F1=0.77 across all models).
  No single feature separates it from ST (Sweeper) or FC (Cutter). This is a
  data limitation, not a modeling failure.
- **3.3% of test pitches are genuinely ambiguous**: no model can classify them
  correctly, and their feature values are indistinguishable from the rest of the test set.
- **XGBoost uniquely resolves 232 borderline pitches** (1.3% of test set) that
  both RF and MLP classify incorrectly, mostly in the SL/FC region.
- **Bimodal feature distributions** in pfx_x and spin_axis are caused by pitcher
  handedness, not noise. The p_throws feature gives models context to interpret
  horizontal movement correctly.

## Data

- Source: MLB Statcast via pybaseball (`statcast()`)
- Season: 2023 (2023-03-30 to 2023-10-01), all MLB pitchers
- 720,684 raw pitches downloaded, sampled to 90,000 (10,000 per class)
- 9 pitch types: FF, SI, FC, SL, ST, CU, KC, CH, FS
- 11 features: physical measurements only, no pitcher identity

## Quickstart

```bash
conda env create -f environment.yml
conda activate baseball_clf

python scripts/fetch_data.py    --config configs/data.yaml
python scripts/preprocess.py    --config configs/data.yaml
python scripts/train.py         --config configs/data.yaml --model configs/model_xgb.yaml
python scripts/train.py         --config configs/data.yaml --model configs/model_rf.yaml
python scripts/train.py         --config configs/data.yaml --model configs/model_dt.yaml
python scripts/train.py         --config configs/data.yaml --model configs/model_knn.yaml
python scripts/train.py         --config configs/data.yaml --model configs/model_mlp.yaml
python scripts/train.py         --config configs/data.yaml --model configs/model_lr.yaml
python scripts/preprocess_specialist.py --config configs/data.yaml
python scripts/train.py         --config configs/data.yaml --model configs/model_xgb_specialist.yaml
python scripts/evaluate.py      --config configs/data.yaml --hierarchical
```

## Project Structure

```
configs/       YAML configs for data pipeline and each model
data/          Raw and processed data (gitignored)
notebooks/     N selection, rule-based baseline, result analysis
scripts/       fetch_data, preprocess, train, evaluate
tests/         Unit tests (49 passing)
results/
  metrics/     Per-model metrics JSON files and predictions CSV
  models/      Saved model pipelines (gitignored)
  figures/     Confusion matrices, heatmaps, bootstrap distributions
docs/          Feature documentation
```

## Key Design Decisions

**No pitcher identity in features**
The model uses only physical pitch measurements plus pitcher handedness
(encoded as 0/1). No pitcher name or ID is included. This forces the
model to learn pitch mechanics rather than pitcher signatures.

**Global sampling**
10,000 rows per pitch type sampled globally across all pitchers. This prevents
any single pitcher from dominating a class and ensures class balance.

**Scaling deferred to train.py**
Tree-based models (RF, XGBoost, DT) do not need scaling. KNN and MLP do.
Applying scaling per-model avoids unnecessary transformation and prevents
test set leakage from a scaler fit on the full dataset.

**HPO via Optuna**
50 trials per model optimizing weighted F1 via 5-fold cross-validation.
Best parameters are saved to JSON and used for the final model refit.

## Notebooks

| Notebook | Purpose |
|---|---|
| `N_selection.ipynb` | Determines n_per_class sampling value from July 2023 data |
| `rule_based_classifier.ipynb` | Hand-crafted baseline with threshold derivation |
| `result_analysis.ipynb` | Full model comparison: significance tests, error analysis, feature distributions |
