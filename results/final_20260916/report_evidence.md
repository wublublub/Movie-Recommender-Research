# Final Experimental Evidence

## Dataset

- Dataset: MovieLens 1M.
- Ratings: 1,000,209.
- Users: 6,040.
- Rated items: 3,706.
- Primary positive interaction: rating >= 4.
- Primary eligible users: 5,984; excluded users: 56.
- Primary chronological split: 463,709 training positives, 55,608 validation positives, and 55,608 test positives.
- Per-user test targets: minimum 2, median 5, maximum 143.
- Rating >= 3 robustness condition: 6,037 eligible users and 836,470 positive interactions.

The split is chronological and uses 10% validation and 10% test holdouts after eligibility filtering. All interpretations concern rating-submission records rather than verified movie-consumption events.

## Experimental Controls

- **Original:** each user's positive interactions remain in recorded pre-test order.
- **Fully shuffled:** each user's complete pre-test sequence is shuffled while users, item multisets, sequence lengths, targets, candidates, and evaluation remain fixed.
- **Recent-context-fixed:** earlier training history is shuffled while the validation/recent-context suffix remains fixed.
- **Static invariance:** Popularity, ItemKNN, and matrix factorization are recomputed under all order conditions to verify that non-sequential models do not change.

The primary comparison is original minus fully shuffled NDCG@10. Five initialization runs and three shuffle seeds are used for the primary condition. Hierarchical 95% confidence intervals resample initialization runs and users within selected runs.

## FPMC Result

Under the primary rating >= 4 Rule A condition, FPMC achieved original NDCG@10 0.074599 and fully shuffled NDCG@10 0.053814. The mean order effect was +0.020784 with hierarchical 95% CI [+0.019412, +0.022146]. This was the largest and most stable order effect among the evaluated sequential models.

## LSTM Result

Under the primary condition, LSTM achieved original NDCG@10 0.059979 and fully shuffled NDCG@10 0.054923. The mean order effect was +0.005057 with hierarchical 95% CI [+0.002268, +0.008866].

The five initialization effects were +0.001574 (seed 760), +0.011911 (761), +0.002946 (762), +0.006008 (763), and +0.002844 (764). All were positive, but variability was material: SD 0.004164 and range +0.001574 to +0.011911. Seed 761 is retained as a valid completed run and is noticeably larger than the other runs; the LSTM magnitude must not be described as uniformly stable.

The five-seed mean original-minus-recent-context-fixed effect was +0.005080 with hierarchical 95% CI [+0.002187, +0.008797]. Recent-context-fixed minus fully shuffled was -0.000024 with hierarchical 95% CI [-0.000511, +0.000441], providing no clear evidence of a difference between those two controls.

## SASRec Result

Under the primary condition, SASRec achieved original NDCG@10 0.055209 and fully shuffled NDCG@10 0.053555. The mean order effect was +0.001654 with hierarchical 95% CI [+0.000506, +0.002669]. The effect was much smaller than FPMC's and smaller than the LSTM mean in this setup.

## Rating Threshold Robustness

The rating >= 3 sensitivity uses seed 760 and is a secondary robustness condition, not a five-seed aggregate. Original-minus-fully-shuffled NDCG@10 was:

- FPMC: +0.035817, 95% CI [+0.032307, +0.039391].
- LSTM: +0.001487, 95% CI [+0.000233, +0.002747].
- SASRec: +0.003794, 95% CI [+0.001559, +0.005659].

Raw metric magnitudes should not be directly compared across thresholds because eligibility and target definitions differ.

## Timestamp Robustness

Rule B orders equal-timestamp records by source order rather than the Rule A `movieId` tie-break. At rating >= 4 and seed 760, original-minus-fully-shuffled NDCG@10 was:

- FPMC: +0.022278, 95% CI [+0.019447, +0.024997].
- LSTM: +0.001617, 95% CI [+0.000120, +0.002918].
- SASRec: +0.001372, 95% CI [+0.000250, +0.002430].

All three effects remain positive under Rule B. Because tie handling changes held-out targets for some users, this is a robustness analysis rather than an identical-target replication.

## Static Baselines

Popularity, ItemKNN, and matrix factorization each passed seven audited runs with maximum absolute order-condition difference exactly 0.0. Their predictions and metrics are invariant to sequence shuffling, as required by the experimental control.

## Supported Conclusions

- Recorded MovieLens 1M rating order contains model-usable signal under this controlled setup.
- FPMC, LSTM, and SASRec respond differently to the same ordering intervention.
- FPMC has the largest and most stable measured order sensitivity in this experiment.
- LSTM has a positive intermediate mean effect but substantial initialization sensitivity.
- SASRec has a smaller positive mean effect.
- Static non-sequential models remain exactly order-invariant.
- These results concern order sensitivity, not universal model superiority or verified viewing chronology.

## Limitations

- Rating-submission timestamps are not movie-viewing timestamps.
- MovieLens 1M is old and specific to one recommendation domain.
- Sequential effect size depends on architecture and experimental configuration.
- Statistical significance is distinct from practical significance.
- The LSTM result has material initialization variability, especially seed 761.
- Five primary initialization runs provide limited run-level inference; seed effects should remain visible.
- Rating-threshold and timestamp-rule analyses are secondary robustness checks and should not be presented as independent confirmatory experiments.
