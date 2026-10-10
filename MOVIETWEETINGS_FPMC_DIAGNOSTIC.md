# MovieTweetings FPMC diagnostic (2026-10-02)

This is a non-scientific diagnostic of the pinned rating-8 protocol. No threshold, FPMC hyperparameter, candidate policy, or target policy was changed. The completed 400-user pilot was not retrained or overwritten. The new 1,500-user sample was selected by the pre-existing fixed-seed eligible-user sampler, independently of outcomes.

## Completed 400-user artifacts

Source: `outputs/movietweetings_experiment/medium_pilot_u400_20261002_v2/`. Original and Fully Shuffled each evaluated 1,260 test events for 400 users. Every user had a non-empty 10-item recommendation list. The saved NDCG and top-10 lists show zero target hits in either condition. The evaluation function rejects nonfinite score vectors, so all 800 evaluated user/condition score vectors passed that check. The saved mapping has 38,013 sorted unique normalized IDs, zero missing targets, and no candidate/target overlap in the order audit. Original recommendations covered 3,388 distinct items, with the most frequent item appearing seven times in 4,000 positions; Shuffled covered 3,426 items, with a maximum frequency of five. Thus top-10 recommendations did not collapse onto a few items.

The 400-user artifacts contain neither FPMC factor checkpoints nor full scores/ranks. Therefore top-20/50/100/500 hits, exact target-rank distributions, score ranges and variances, and validation ranks **cannot be recovered from those artifacts without retraining**. They are not imputed. The only available 400-user validation ranking statistic is the selected Original NDCG@10 of 0.000684, proving at least one validation top-10 hit.

Global-training-visible target events numbered 986/1,260 (78.25%) in validation and 957/1,260 (75.95%) in test; cold events numbered 274 and 303. All 400-user FPMC test warm- and cold-target top-10 event hit rates and NDCG contributions were zero in both conditions. Since the raw audit found no repeated user/movie pairs, all held-out positive targets were user-unseen. Training-item frequency for validation versus test targets was 7.62 versus 7.18 on average, median four for both; targets of frequency zero or one comprised 33.89% versus 36.03%. Median prior-positive history length was 28 versus 31. Test candidate counts spanned 37,321–38,006 (median 37,984), identical across order conditions. The test target set is slightly colder and rarer, but unavailable 400-user ranks prevent attributing the zero specifically to this shift.

## Implementation checks

The sorted normalized metadata IDs index all four FPMC factor tables. `np.searchsorted` maps historical and target IDs into the same catalog. All 38,013 IDs are valid table rows, and no pilot target falls outside the mapping. FPMC iterates over positive transitions, updating user, positive-item, previous-item, and next-item factors; negatives are sampled from the full catalog while avoiding a user's seen sequence. The candidate construction removes previously seen items except held-out targets, with no target/exclusion overlap. Unit tests verify positive and transition updates, finite nonconstant scores, valid ranks, and equivalence of batched FPMC scoring to the original per-user scorer. No implementation or indexing defect was found. This is not a claim that the model is optimally tuned.

## Predeclared 1,500-user pilot

Command: `PYTHONPATH=src .venv/Scripts/python.exe -m movie_recommender.movietweetings_experiment --pilot-users 1500 --name diagnostic_pilot_u1500_20261002` (PowerShell used `$env:PYTHONPATH='src'`). Output: `outputs/movietweetings_experiment/diagnostic_pilot_u1500_20261002/`. The run used 36,842 training, 4,971 validation, and 4,971 test positive events; 38,013 catalog items; initialization 760; shuffle 20760; Original-validation-only selection. All six model/condition cells completed with zero order-audit mismatches. FPMC selected the unchanged 24-factor, 3-epoch configuration with transition weight 1.0; validation NDCG@10 was 0.014267.

| Model | Original test NDCG@10 | Fully Shuffled test NDCG@10 |
|---|---:|---:|
| FPMC | 0.018391 | 0.000158 |
| LSTM | 0.025235 | 0.023710 |
| SASRec | 0.022819 | 0.020771 |

FPMC Original had 83 target events at ranks 1–10, 104 at 1–20, 130 at 1–50, 153 at 1–100, and 278 at 1–500. Fully Shuffled had 1, 7, 15, 27, and 116 respectively. These larger-K counts are diagnostics, not alternative primary metrics. Original FPMC user-level HR@10 was 0.052 (78/1,500 users); Shuffled was 0.000667 (1/1,500). Every scored candidate vector was finite and had nonzero standard deviation. Original global candidate scores ranged from −0.1621 to 0.4895, with median within-vector standard deviation 0.01927; Shuffled ranged from −0.1167 to 0.1438, median standard deviation 0.01878. Diagnostic rank<=10 agreed with saved top-10 membership for all 4,971 targets in each condition.

Original target-rank min/P25/median/P75/P95 was 1/6,577.5/16,708/27,351.5/35,929; Shuffled was 5/7,639.5/17,696/28,023/36,033. Original recommendations covered 8,046 distinct items across 15,000 positions; the most frequent item occurred 230 times. Shuffled covered 8,675, with a maximum of 86. There is some concentration in Original but no single-item collapse.

The 1,500-user test set had 4,381 warm and 590 cold target events. Original warm targets had 82 top-10 hits (event HR@10 0.01872; single-target discount-NDCG@10 0.01165), versus one cold hit (HR 0.001695; discount-NDCG 0.000535). Shuffled warm targets had one hit (HR 0.000228; discount-NDCG 0.0000883); cold targets had zero. These event-level single-target diagnostics are **not** the primary per-user NDCG@10. Original warm/cold median ranks were 16,287/18,754; Shuffled 17,306/20,508. Cold targets remain in the primary evaluation.

Validation and test each had 4,971 events. Their warm proportions were 88.05% and 88.13%; median training-item frequency was 14 for both, means 28.47 and 26.77; median prior-positive history lengths were 32 and 35. The Original validation target-rank median was 16,176 (68 top-10 target hits), versus test median 16,708 (83 hits). Test is not categorically harder at this scale. The change from zero FPMC hits at 400 users is consistent with sparse-hit/full-catalog behavior and a much larger, warmer global training pool—not proof that sample size alone caused it, because the deterministic 400- and 1,500-user samples are different.

## Multi-initialization and run cost

The runner now accepts `--initialization-seeds 760,761,762,763,764` and defaults to seed 760 when omitted. Each seed tunes on Original validation only, freezes selected settings across its Original and shuffle conditions, and has separate result and selection files. Per-user rows, per-seed effects, model runtimes, protocol, and completion metadata identify seeds. A separate 10-user, two-initialization integration run completed 120 expected rows (2 seeds × 3 models × 2 conditions × 10 users), with distinct neural checkpoints and no missing Cartesian cells. It is not a scientific result.

The 1,500-user pilot took 188.5 seconds wall time; model tuning plus condition times were FPMC 20.4, SASRec 77.8, and LSTM 50.8 seconds. Peak process RSS was 1.50 GiB. Scaling 36,842 pilot training events to 275,930 formal training events and two pilot conditions to seven formal conditions gives about 0.76 model-hour per initialization, or 3.8 model-hours for five initializations. If LSTM uses five rather than the pilot-selected three epochs, the five-seed model estimate rises to about 4.5 hours. Shared preprocessing, static baselines (not run in the pilot), output I/O, and nonlinear scaling are **not** included; total formal wall time cannot be reliably bounded from this pilot alone.

The earlier completion-metadata failure was Git's sandbox ownership check on `git branch --show-current`, after scientific outputs had been written. The command now supplies a repository-local `safe.directory` option, without changing training or evaluation; the completed 400-user repeat, 1,500-user pilot, and two-seed integration each wrote completion metadata. The original partial pilot directory remains untouched.
