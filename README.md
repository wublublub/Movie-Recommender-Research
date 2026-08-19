# MovieLens 1M Recorded-Order Experiments

This COMPSCI 760 project asks:

> To what extent does the recorded MovieLens rating-submission order contribute to the recommendation performance of sequential recommender systems?

MovieLens timestamps record **rating submission**, not confirmed movie viewing. Results therefore concern the order of positive rating events and must not be described as effects of true viewing order.

## Primary workflow

`ml1m_order_experiment.py` is the primary controlled experiment. It uses a deterministic chronological train/validation/test split, multi-target full-catalog ranking, matched FPMC and SASRec controls, independent random-seed roles, and paired user-level inference.

The experiment compares:

1. **Original order:** all positive train and validation interactions in recorded pre-test order.
2. **Fully shuffled pre-test:** independently shuffles every user's complete train-plus-validation history while preserving users, item multisets, lengths, item frequencies, validation targets, and test targets. This is the primary intervention.
3. **Recent-context-fixed shuffle:** shuffles the earlier training history but preserves the validation/recent-context suffix. This is a secondary ablation.

Popularity, ItemKNN, and TruncatedSVD matrix factorization are order-invariant controls. FPMC and SASRec are evaluated under every order condition with common users, mappings, targets, candidates, metrics, and tuning decisions.

`ml1m_revised_sequence_feasibility.py` and `ml1m_model_feasibility.py` remain available as legacy feasibility workflows. Their historical outputs should not be mixed directly with the controlled experiment.

## Repository structure

- `data/raw/ml-1m/`: immutable MovieLens 1M files; ignored by Git
- `src/movie_recommender/`: loaders, legacy analyses, and controlled experiment code
- `tests/`: unit, invariant, causal-attention, synthetic end-to-end, and reproducibility tests
- `outputs/`: generated experiment artifacts; ignored by Git
- `notebooks/`: optional exploratory work, not the source of formal results

## MovieLens 1M setup

Place the official MovieLens 1M files at:

```text
data/raw/ml-1m/ratings.dat
data/raw/ml-1m/movies.dat
data/raw/ml-1m/users.dat
```

Raw data must not be edited or committed. Each run records the SHA-256 checksum of `ratings.dat`.

## Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The pinned requirements reflect the tested Windows/Python 3.12 environment. PyTorch automatically uses CUDA when available; the current local installation is CPU-only.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Non-final smoke test

Fast outputs are explicitly labelled non-final:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m movie_recommender.ml1m_order_experiment `
  --fast-test `
  --output-dir outputs/ml1m_order_experiment_smoke
```

## Formal experiment

The default formal run uses three deterministic sequence-shuffle seeds. Additional initialization seeds can be supplied as a comma-separated list when compute permits.

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m movie_recommender.ml1m_order_experiment `
  --output-dir outputs/ml1m_order_experiment_rating4
```

Example initialization-seed sensitivity:

```powershell
--model-initialization-seeds 760,761,762
```

Relevance sensitivity uses `--relevance-threshold 3`; `--all-ratings` treats every rating event as an interaction. The main interpretation remains positive rating-event order for the default threshold of 4.

## Robustness matrix

`ml1m_robustness.py` runs a resumable matrix with five rating>=4 initialization seeds, a rating>=3 sensitivity run, and the alternative `Timestamp, source order` tie rule. It writes `run_manifest.csv` before and after every cell, never overwrites completed directories, and aggregates only runs marked complete.

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m movie_recommender.ml1m_robustness `
  --runs-dir outputs/ml1m_order_robustness_runs_20260817 `
  --aggregate-dir outputs/ml1m_order_robustness_20260817 `
  --primary-result-dir outputs/ml1m_order_experiment_rating4_20260817 `
  --initialization-seeds 760,761,762,763,764
```

To resume incomplete cells, rerun the same command. To rebuild only the cross-run tables and report:

```powershell
.\.venv\Scripts\python.exe -m movie_recommender.ml1m_robustness `
  --runs-dir outputs/ml1m_order_robustness_runs_20260817 `
  --aggregate-dir outputs/ml1m_order_robustness_20260817 `
  --primary-result-dir outputs/ml1m_order_experiment_rating4_20260817 `
  --initialization-seeds 760,761,762,763,764 `
  --aggregate-only
```

The aggregate uses a two-level bootstrap that samples initialization runs and then users within selected runs. Secondary threshold, timestamp, and recent-context comparisons are labelled exploratory or robustness analyses rather than confirmatory tests.

## Outputs

Each result directory is self-contained and includes configuration, environment and hardware metadata, dataset checksum, split audit, eligible/excluded users, candidate audits, validation results, selected hyperparameters, per-user and aggregate metrics, seed-level results, paired differences, statistical comparisons, training curves, sensitivity summaries, figures, and an actual-results Markdown report.

The cross-run robustness directory contains only manifests and aggregate evidence tables; raw per-run artifacts remain in their separate result directories.

On the current 16-thread CPU-only machine, smoke tests take roughly a minute. Formal runtime depends strongly on initialization seeds and epochs; allow tens of minutes and consult `environment_and_hardware.json` for the measured runtime.

Smoke outputs and subgroup analyses are exploratory. Only a completed formal result directory should be used for confirmatory claims, and final paper claims still require checking robustness across relevance thresholds, history requirements, and model initialization seeds.
