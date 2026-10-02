import copy

import numpy as np
import pandas as pd
import pytest
import torch

from movie_recommender.lstm_model import LSTMRecommender
from movie_recommender.ml1m_lstm_experiment import make_history_batch, make_padded_batch
from movie_recommender.ml1m_model_feasibility import FPMC, build_interaction_matrix, train_item_knn
from movie_recommender.ml1m_revised_sequence_feasibility import (
    RevisedConfig, SASRec, SASRecParameters, build_ranking_sets,
    pad_sequences, revised_chronological_split, set_ranking_metrics,
)
from movie_recommender.movietweetings_experiment import (
    block_item_knn, build_parser, compact_sets, condition_grid, current_git_branch,
    evaluate, fpmc_rank_diagnostics, fpmc_scorer, lstm_valid_logits, neural_scorer,
    parse_initialization_seeds, run, seed_metadata, summarize, top_metrics,
)


@pytest.mark.parametrize("tied", [False, True])
def test_top_metrics_exactly_match_existing_evaluation(tied):
    rng = np.random.default_rng(760)
    movie_ids = np.arange(100, 1100) * 3
    for _ in range(20):
        scores = rng.normal(size=1000)
        if tied:
            scores = scores.round(0)
        excluded = np.sort(rng.choice(800, 40, replace=False))
        targets = np.arange(880, 905)
        candidates = np.setdiff1d(np.arange(1000), excluded)
        actual, top = top_metrics(scores, excluded, targets)
        expected, expected_top = set_ranking_metrics(scores[candidates], movie_ids[candidates], set(movie_ids[targets]))
        assert actual == pytest.approx(expected)
        np.testing.assert_array_equal(movie_ids[top], expected_top)


def test_all_tied_scores_exclude_seen_items():
    metrics, top = top_metrics(np.zeros(30), np.arange(5), np.array([5, 6]))
    np.testing.assert_array_equal(top, np.arange(5, 15))
    assert metrics["recall_10"] == 1


def test_compact_candidates_match_existing_including_negative_history():
    rows = [(u, i + u * 20, 8 if i % 3 else 1, 1500000000 + i // 2) for u in (1, 2) for i in range(18)]
    frame = pd.DataFrame(rows, columns=["UserID", "MovieID", "Rating", "Timestamp"])
    split = revised_chronological_split(frame, RevisedConfig(relevance_threshold=8))
    ids = np.arange(1, 80)
    for targets in (split.validation_positive, split.test_positive):
        reference, _ = build_ranking_sets(split, targets, ids)
        compact = compact_sets(split, targets, ids)
        for row, user in enumerate(compact.user_ids):
            np.testing.assert_array_equal(np.delete(ids, compact.excluded[row]), reference[user].candidate_movie_ids)
            assert set(ids[compact.targets[row]]) == reference[user].relevant_movie_ids
        result = evaluate(compact, lambda a, b: np.zeros((b-a, len(ids))), batch_size=1)
        assert len(result) == 2


def test_batched_knn_matches_original():
    rng = np.random.default_rng(18)
    matrix = build_interaction_matrix([rng.choice(40, 15, replace=False) for _ in range(50)], 40)
    np.testing.assert_allclose(block_item_knn(matrix, 8, 7).toarray(), train_item_knn(matrix, 8).toarray(), atol=1e-6)


def test_lstm_projection_optimization_preserves_loss_gradients_and_scores():
    torch.manual_seed(760)
    model = LSTMRecommender(40, 32, 32)
    reference = copy.deepcopy(model)
    sequences = [np.array([0, 1, 2, 3]), np.array([4, 5])]
    inputs, targets, lengths = make_padded_batch(sequences, 8)
    original_loss = torch.nn.functional.cross_entropy(reference(inputs).reshape(-1, 40), targets.reshape(-1), ignore_index=-100)
    logits, selected_targets = lstm_valid_logits(model, inputs[:, :int(lengths.max())], targets[:, :int(lengths.max())])
    new_loss = torch.nn.functional.cross_entropy(logits, selected_targets)
    torch.testing.assert_close(new_loss, original_loss)
    new_loss.backward()
    original_loss.backward()
    for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, expected.grad, atol=1e-7, rtol=1e-5)
    history, history_lengths = make_history_batch(sequences, 8)
    with torch.no_grad():
        expected_scores = reference.score_last(history, history_lengths).numpy()
    np.testing.assert_allclose(neural_scorer(model, sequences, "lstm", 8)(0, 2), expected_scores, atol=1e-7)


def test_nonfinite_scores_rejected():
    with pytest.raises(ValueError, match="Nonfinite"):
        top_metrics(np.full(20, np.nan), np.array([], dtype=int), np.array([1]))


def test_two_layer_sasrec_left_padding_has_finite_scores_and_restores_backend():
    torch.manual_seed(760)
    model = SASRec(40, SASRecParameters(64, 100, 2, 4, .2, .0005))
    before = torch.backends.mha.get_fastpath_enabled()
    scorer = neural_scorer(model, [np.arange(5), np.arange(11)], "sasrec", 100)
    scores = scorer(0, 2)
    assert scores.shape == (2, 40)
    assert np.isfinite(scores).all()
    assert torch.backends.mha.get_fastpath_enabled() == before


def test_safe_sasrec_inference_matches_nonfused_training_path_without_dropout():
    torch.manual_seed(760)
    model = SASRec(40, SASRecParameters(64, 100, 2, 4, 0., .0005))
    sequences = [np.arange(5), np.arange(11)]
    model.train()
    with torch.no_grad():
        reference = model.score_all(torch.from_numpy(pad_sequences(sequences, 100))).numpy()
    actual = neural_scorer(model, sequences, "sasrec", 100)(0, 2)
    np.testing.assert_allclose(actual, reference, atol=1e-6, rtol=1e-5)


def test_batched_fpmc_scores_match_existing_model():
    sequences = [np.arange(5), np.arange(8)]
    model = FPMC(2, 40, 24, .03, .002, 760, 17760)
    actual = fpmc_scorer(model, sequences, .5)(0, 2)
    reference = np.stack([model.score(u, int(s[-1]), np.arange(40), .5) for u, s in enumerate(sequences)])
    np.testing.assert_allclose(actual, reference, atol=1e-7, rtol=1e-5)


def test_pilot_limit_rejects_smoke_and_large_runs_before_writing() -> None:
    with pytest.raises(ValueError, match="Pilot requires"):
        run(smoke=True, pilot_users=400)
    with pytest.raises(ValueError, match="Pilot requires"):
        run(pilot_users=1501)


def test_primary_only_summary_does_not_require_recent_fixed(tmp_path) -> None:
    rows = []
    for model_name in ("fpmc", "lstm", "sasrec"):
        for user_id in (1, 2):
            for condition, ndcg in (("original_order", 0.2),
                                    ("fully_shuffled_pretest", 0.1)):
                rows.append({"user_id": user_id, "model_name": model_name,
                             "order_condition": condition, "ndcg_10": ndcg,
                             "model_initialization_seed": 760,
                             "sequence_shuffle_seed": 20760 if condition != "original_order" else np.nan,
                             "precision_10": 0.0, "recall_10": 0.0,
                             "hit_rate_10": 0.0, "mrr_10": 0.0, "map_10": 0.0})

    _, effects = summarize(pd.DataFrame(rows), tmp_path, False,
                           ("fully_shuffled_pretest",))

    assert len(effects) == 3
    assert effects["mean_paired_difference"].tolist() == pytest.approx([0.1] * 3)


def test_initialization_seed_cli_and_cartesian_grid() -> None:
    assert build_parser().parse_args([]).initialization_seeds is None
    assert parse_initialization_seeds(None) == [760]
    assert build_parser().parse_args(["--initialization-seeds", "760,761"]).initialization_seeds == [760, 761]
    with pytest.raises(ValueError, match="distinct"):
        parse_initialization_seeds("760,760")
    with pytest.raises(ValueError, match="comma-separated"):
        parse_initialization_seeds("760,nope")
    grid = condition_grid([760, 761], [20760, 20761], True)
    assert len(grid) == 6
    assert set(grid) == {
        (760, "original_order", None), (761, "original_order", None),
        (760, "fully_shuffled_pretest", 20760), (760, "fully_shuffled_pretest", 20761),
        (761, "fully_shuffled_pretest", 20760), (761, "fully_shuffled_pretest", 20761),
    }
    formal_grid = condition_grid([760, 761], [20760, 20761, 20762], False)
    assert len(formal_grid) == 14
    assert len(set(formal_grid)) == 14
    assert all(sum(row[0] == seed for row in formal_grid) == 7 for seed in (760, 761))
    assert seed_metadata([760, 761], [20760, 20761]) == {
        "initialization_seeds": [760, 761], "shuffle_seeds": [20760, 20761],
    }


def test_multi_initialization_summary_pairs_within_seed(tmp_path) -> None:
    rows = []
    for model_name in ("fpmc", "lstm", "sasrec"):
        for initialization_seed in (760, 761):
            for user_id in (1, 2):
                for condition, ndcg in (("original_order", 0.2 + initialization_seed - 760),
                                        ("fully_shuffled_pretest", 0.1 + initialization_seed - 760)):
                    rows.append({"user_id": user_id, "model_name": model_name,
                                 "order_condition": condition, "ndcg_10": ndcg,
                                 "model_initialization_seed": initialization_seed,
                                 "sequence_shuffle_seed": 20760 if condition != "original_order" else np.nan,
                                 "precision_10": 0., "recall_10": 0., "hit_rate_10": 0.,
                                 "mrr_10": 0., "map_10": 0.})
    _, effects = summarize(pd.DataFrame(rows), tmp_path, False, ("fully_shuffled_pretest",))
    assert effects.mean_paired_difference.tolist() == pytest.approx([0.1] * 3)
    seed_effects = pd.read_csv(tmp_path / "shuffle_seed_effects.csv")
    assert set(seed_effects.initialization_seed) == {760, 761}
    assert len(seed_effects) == 6
    by_initialization = pd.read_csv(tmp_path / "aggregate_metrics_by_initialization.csv")
    assert len(by_initialization) == 12
    assert set(by_initialization.model_initialization_seed) == {760, 761}
    assert effects.initialization_count.tolist() == [2, 2, 2]
    assert effects.uncertainty.str.contains("hierarchical").all()


def test_git_completion_metadata_uses_local_safe_directory(monkeypatch) -> None:
    def fake_check_output(command, **kwargs):
        from movie_recommender.movietweetings_experiment import ROOT
        assert command[1:3] == ["-c", f"safe.directory={ROOT.as_posix()}"]
        assert command[-2:] == ["branch", "--show-current"]
        return "codex/movietweetings-integration\n"

    monkeypatch.setattr("movie_recommender.movietweetings_experiment.subprocess.check_output", fake_check_output)
    assert current_git_branch() == "codex/movietweetings-integration"


def test_fpmc_training_and_rank_diagnostics_update_factors() -> None:
    sequences = [np.array([0, 1, 2, 3]), np.array([4, 5, 6, 7])]
    model = FPMC(2, 20, 4, .03, .002, 760, 17760)
    preference_before = model.item_preference_factors.copy()
    transitions_before = model.next_item_factors.copy()
    model.fit(sequences, 1)
    assert np.any(model.item_preference_factors[[1, 2, 3, 5, 6, 7]] != preference_before[[1, 2, 3, 5, 6, 7]])
    assert np.any(model.next_item_factors[[1, 2, 3, 5, 6, 7]] != transitions_before[[1, 2, 3, 5, 6, 7]])
    sets = type("Sets", (), {"movie_ids": np.arange(20), "user_ids": np.array([10, 11]),
                            "excluded": [np.array([0]), np.array([4])],
                            "targets": [np.array([3]), np.array([7])]})()
    diagnostics = fpmc_rank_diagnostics(sets, fpmc_scorer(model, sequences, 1.), {3})
    assert len(diagnostics) == 2
    assert diagnostics.all_candidate_scores_finite.all()
    assert diagnostics.candidate_score_std.gt(0).all()
    assert diagnostics.warm_in_global_training.tolist() == [True, False]
    assert diagnostics["rank"].between(1, 19).all()


def test_fpmc_rank_diagnostics_respects_exclusion_and_id_ties() -> None:
    movie_ids = np.arange(10, 22)
    sets = type("Sets", (), {"movie_ids": movie_ids, "user_ids": np.array([7]),
                            "excluded": [np.array([0])], "targets": [np.array([2])]})()
    scores = np.array([[100., 6., 5., 5., 0., 0., 0., 0., 0., 0., 0., 0.]])
    diagnostics = fpmc_rank_diagnostics(sets, lambda start, end: scores[start:end], {12})
    assert diagnostics.target_movie_id.tolist() == [12]
    assert diagnostics["rank"].tolist() == [2]
    assert diagnostics.candidate_count.tolist() == [11]
