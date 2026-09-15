# Final MovieLens 1M Order-Sensitivity Results

## Research question

To what extent does recorded MovieLens 1M rating-submission order provide useful sequential recommendation signal when the same user-item interactions are retained but their ordering is changed?

MovieLens timestamps represent rating-submission records, not confirmed movie-consumption times. These results therefore concern sensitivity to recorded rating order, not true viewing sequence.

## Experimental design

The controlled experiment keeps users, items, held-out targets, candidate sets, evaluation, and model-selection decisions fixed while changing only pre-test interaction order:

- **Original order:** recorded chronological order, with deterministic tie handling.
- **Fully shuffled order:** each user's full pre-test history is shuffled while preserving its item multiset and length.
- **Recent-context-fixed control:** earlier training history is shuffled while the recent validation suffix is preserved.

The primary condition uses ratings >= 4, timestamp Rule A (`timestamp, movieId, source order`), five initialization seeds (760-764), three shuffle seeds (20760-20762), and NDCG@10. Hierarchical confidence intervals resample initialization runs and users within runs.

## Final primary result

| Model | Original NDCG@10 | Shuffled NDCG@10 | Difference | Hierarchical 95% CI |
| --- | ---: | ---: | ---: | ---: |
| FPMC | 0.074599 | 0.053814 | +0.020784 | +0.019412 to +0.022146 |
| LSTM | 0.059979 | 0.054923 | +0.005057 | +0.002268 to +0.008866 |
| SASRec | 0.055209 | 0.053555 | +0.001654 | +0.000506 to +0.002669 |

FPMC shows the largest and most stable order sensitivity in this setup. LSTM shows a positive intermediate effect in all five initialization runs, but the magnitude is materially initialization-sensitive (SD 0.004164; range +0.001574 to +0.011911), with seed 761 noticeably larger than the others. SASRec shows a much smaller effect. Popularity, ItemKNN, and matrix factorization remain exactly invariant to shuffling.

These findings show that FPMC, LSTM, and SASRec react differently to the same ordering intervention. They do not establish a universal architecture ranking, and absolute recommendation performance must be distinguished from order sensitivity.

## Package contents

- `model_comparison.csv`: primary five-seed FPMC, LSTM, and SASRec comparison.
- `lstm_per_initialization_results.csv`: transparent LSTM seed-level results, uncertainty, recent-context control, and run metadata.
- `robustness_conditions.csv`: primary, rating-threshold, timestamp-rule, and recent-context comparisons.
- `static_baseline_audit.csv`: exact order-invariance audit for non-sequential baselines.
- `report_evidence.md`: report-ready factual summary and supported conclusions.
- `provenance.json`: source locations, source checksums, and confirmation that smoke outputs were excluded.
- `figures/`: four report-ready figures copied from the authoritative aggregate output.

## Provenance and reproduction

This package was generated from `outputs/ml1m_lstm_robustness_20260916`, which combines the completed September LSTM matrix with the preserved August FPMC/SASRec manifest. Full raw outputs remain in their original ignored directories and were not moved or modified. The compact tables were programmatically checked against the authoritative aggregate CSV files.

See the repository-level `README.md` for environment setup, test, formal execution, and aggregate-only commands.
