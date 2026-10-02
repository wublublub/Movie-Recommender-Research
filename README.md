# MovieLens 1M Recorded-Order Experiments

This COMPSCI 760 project asks:

> To what extent does the recorded MovieLens rating-submission order contribute to the recommendation performance of sequential recommender systems?

MovieLens timestamps record **rating submission**, not confirmed movie viewing. Results therefore concern the order of positive rating events and must not be described as effects of true viewing order.

## Primary workflow

`ml1m_order_experiment.py` is the primary controlled experiment. It uses a deterministic chronological train/validation/test split, multi-target full-catalog ranking, matched FPMC, LSTM, and SASRec controls, independent random-seed roles, and paired user-level inference.

The experiment compares:

1. **Original order:** all positive train and validation interactions in recorded pre-test order.
2. **Fully shuffled pre-test:** independently shuffles every user's complete train-plus-validation history while preserving users, item multisets, lengths, item frequencies, validation targets, and test targets. This is the primary intervention.
3. **Recent-context-fixed shuffle:** shuffles the earlier training history but preserves the validation/recent-context suffix. This is a secondary ablation.

Popularity, ItemKNN, and TruncatedSVD matrix factorization are order-invariant controls. FPMC, LSTM, and SASRec are evaluated under every order condition with common users, mappings, targets, candidates, metrics, and tuning decisions. `--models` can select a subset without changing the shared experiment semantics.

`ml1m_revised_sequence_feasibility.py` and `ml1m_model_feasibility.py` remain available as legacy feasibility workflows. Their historical outputs should not be mixed directly with the controlled experiment.

## Repository structure

- `data/raw/ml-1m/`: immutable MovieLens 1M files; ignored by Git
- `src/movie_recommender/`: loaders, legacy analyses, and controlled experiment code
- `tests/`: unit, invariant, causal-attention, synthetic end-to-end, and reproducibility tests
- `outputs/`: generated experiment artifacts; ignored by Git
- `results/final_20260916/`: compact, tracked final tables, figures, provenance, and report evidence
- `notebooks/`: optional exploratory work, not the source of formal results

## Experiment design

The frozen formal study uses MovieLens 1M, chronological per-user holdouts, and NDCG@10 as the primary metric. Ratings >= 4 define the primary positive interactions; ratings >= 3 provide a robustness condition. Rule A orders equal-timestamp events by `timestamp, movieId, source order`, while Rule B uses `timestamp, source order`.

The non-sequential controls are Popularity, ItemKNN, and TruncatedSVD matrix factorization. The sequential models are FPMC, LSTM, and SASRec. Original, fully shuffled, and recent-context-fixed histories share users, items, targets, candidates, evaluation, and tuning decisions so that the principal intervention is pre-test order.

## Final results

The completed primary rating >= 4 Rule A condition uses five initialization seeds (760-764) and three shuffle seeds (20760-20762):

| Model | Original NDCG@10 | Shuffled NDCG@10 | Difference | Hierarchical 95% CI |
| --- | ---: | ---: | ---: | ---: |
| FPMC | 0.074599 | 0.053814 | +0.020784 | +0.019412 to +0.022146 |
| LSTM | 0.059979 | 0.054923 | +0.005057 | +0.002268 to +0.008866 |
| SASRec | 0.055209 | 0.053555 | +0.001654 | +0.000506 to +0.002669 |

FPMC shows the largest and most stable order sensitivity in this setup. LSTM is positive in all five runs but materially initialization-sensitive (SD 0.004164; range +0.001574 to +0.011911), with seed 761 noticeably larger. SASRec shows a smaller effect. All static baselines remain exactly invariant.

These are architecture-specific responses to the same intervention, not a universal ranking of recommender architectures. MovieLens timestamps record rating submission rather than confirmed movie consumption. The authoritative compact evidence package is `results/final_20260916/`.

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

The default formal runner uses three deterministic sequence-shuffle seeds. The experiment is now frozen; these commands document reproduction rather than invite retuning.

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m movie_recommender.ml1m_order_experiment `
  --output-dir outputs/ml1m_order_experiment_rating4
```

Example initialization-seed specification:

```powershell
--model-initialization-seeds 760,761,762
```

Relevance sensitivity uses `--relevance-threshold 3`; `--all-ratings` treats every rating event as an interaction. The main interpretation remains positive rating-event order for the default threshold of 4.

## Integrated LSTM validation

The compatibility command and direct model selection both use the same authoritative runner and output schema. Smoke metrics are non-scientific.

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m movie_recommender.ml1m_order_experiment `
  --models lstm `
  --fast-test `
  --output-dir outputs/ml1m_lstm_integrated_smoke
```

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

The completed FPMC/SASRec directories remain immutable. The completed LSTM matrix used the same five initialization seeds and three shuffle seeds while writing only to new directories:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m movie_recommender.ml1m_robustness `
  --models lstm `
  --runs-dir outputs/ml1m_lstm_robustness_runs_20260916 `
  --aggregate-dir outputs/ml1m_lstm_robustness_20260916 `
  --primary-result-dir outputs/ml1m_lstm_order_experiment_rating4_init760_20260916 `
  --initialization-seeds 760,761,762,763,764 `
  --include-manifest outputs/ml1m_order_robustness_20260817/run_manifest.csv
```

Including the preserved manifest produces unified `model_comparison.csv` and robustness tables without rerunning or rewriting the completed FPMC/SASRec directories.

To regenerate only the unified aggregate after all cells are complete, append `--aggregate-only` to the same command. Do not rerun formal cells unless a genuine reproducibility defect is discovered.

## Outputs

Each result directory is self-contained and includes configuration, environment and hardware metadata, dataset checksum, split audit, eligible/excluded users, candidate audits, validation results, selected hyperparameters, per-user and aggregate metrics, seed-level results, paired differences, statistical comparisons, training curves, sensitivity summaries, figures, and an actual-results Markdown report.

The cross-run robustness directory contains only manifests and aggregate evidence tables, including `model_comparison.csv`; raw per-run artifacts remain in their separate result directories. Full raw outputs are intentionally ignored by Git. The small `results/final_20260916/` package is version-controlled so teammates can inspect the final findings without copying large per-user files.

On the current 16-thread CPU-only machine, smoke tests take roughly a minute. Formal runtime depends strongly on initialization seeds and epochs; allow tens of minutes and consult `environment_and_hardware.json` for the measured runtime.

Smoke outputs and subgroup analyses are exploratory and are excluded from the final package. The formal robustness matrix is complete: all seven LSTM cells passed, the preserved FPMC/SASRec results were incorporated without modification, and no additional modelling or post-hoc tuning is planned.

## Foursquare NYC adapter

The TSMC2014 NYC integration reads the original Latin-1, tab-separated file from `data/raw/dataset_tsmc2014/dataset_TSMC2014_NYC.txt`. It preserves repeated user-venue events, sorts by UTC timestamp with source-row order as the deterministic tie-break, and temporarily maps canonical `user_id`, `item_id`, and `timestamp` fields to the shared runner's legacy column names.

Foursquare uses full-catalog candidates including previously visited venues. Held-out targets retain event multiplicity: metrics average single-target Top-K outcomes across every held-out check-in event, so repeated visits are not collapsed. This differs intentionally from MovieLens's unseen-item, set-target policy.

Run only the bounded pilot until a formal run is explicitly approved:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m movie_recommender.foursquare_order_experiment `
  --data-dir data/raw/dataset_tsmc2014 `
  --output-dir outputs/foursquare_nyc_pilot `
  --fast-test `
  --fast-user-limit 12 `
  --maximum-epochs 1 `
  --batch-size 8 `
  --lstm-batch-size 2
```

The non-fast command is documented by `--help` but should not be launched without explicit approval. Parameter candidates are selected on original-order validation only and then frozen for original and shuffled test conditions.

Use `--user-limit N` for a deterministic medium-scale subset without enabling the architecture and epoch shortcuts associated with `--fast-test`.
