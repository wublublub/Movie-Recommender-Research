# Compact final cross-dataset results

These are small, versioned copies of completed result tables. They are not replacement inputs for the runners, and the large per-user metrics, raw datasets, and checkpoints remain Git-ignored.

- `cross_dataset_comparison_formal.csv`: copied from `outputs/movietweetings_experiment/formal_rating8_5init_20261002/cross_dataset_comparison_formal.csv`; its `source` column identifies the three frozen formal studies.
- `matched_aggregate_effects.csv`, `matched_bootstrap_intervals.csv`, `matched_structural_summary.csv`, and `matched_integrity_validation.json`: copied from the corresponding files in `outputs/movietweetings_matched_targeted_formal_20261003/` (remove the `matched_` filename prefix).

The matched-targeted primary population is 9,287 users. Effects are Original minus perturbed NDCG@10. `broad_minus_local` is Broad minus Local NDCG@10; a positive value indicates the Local perturbation reduced performance more. Its intervals use 5,000 paired hierarchical bootstrap replicates over users, initialization seeds, and perturbation seeds. Do not compare absolute NDCG across datasets as recommendation quality: target and candidate policies differ.

The archived formal output directories and their `run_config.json`, protocol, hashes, validation, and seed-level tables are the complete provenance. See the root `README.md` for setup and reproduction entry points. Never overwrite archived experiment directories.
