import numpy as np
import pandas as pd
import pytest

from movie_recommender.movietweetings_kendall_calibration import (
    broad_budget_indices, local_budget_indices, local_window_size,
    normalized_kendall_distance,
    structurally_eligible_users,
)


@pytest.mark.parametrize(
    ("permutation", "expected"),
    [([1, 2, 3, 4], 0), ([4, 3, 2, 1], 1),
     ([2, 1, 3, 4], 1 / 6), ([3, 1, 4, 2], 3 / 6)],
)
def test_exact_normalized_kendall_examples(permutation: list[int], expected: float) -> None:
    assert normalized_kendall_distance(
        np.array([1, 2, 3, 4]), np.array(permutation)
    ) == pytest.approx(expected)


def test_kendall_rejects_mismatched_or_repeated_items() -> None:
    with pytest.raises(ValueError, match="multisets"):
        normalized_kendall_distance(np.array([1, 2, 3]), np.array([1, 2, 4]))
    with pytest.raises(ValueError, match="unique"):
        normalized_kendall_distance(np.array([1, 1, 2]), np.array([1, 1, 2]))


@pytest.mark.parametrize("target", [0.05, 0.10, 0.15, 0.20])
def test_local_budget_is_deterministic_window_limited_and_exact(target: float) -> None:
    original = np.arange(16)
    first = local_budget_indices(16, target, 20760)
    second = local_budget_indices(16, target, 20760)
    assert np.array_equal(first, second)
    assert np.array_equal(np.sort(first), original)
    for start in range(0, 16, 4):
        assert set(first[start:start + 4]) == set(range(start, start + 4))
    assert abs(normalized_kendall_distance(original, first) - target) <= 0.01


def test_long_histories_use_eight_item_local_windows() -> None:
    assert local_window_size(32) == 4
    assert local_window_size(33) == 8
    original = np.arange(64)
    indices = local_budget_indices(64, 0.05, 20760)
    for start in range(0, len(original), 8):
        assert set(indices[start:start + 8]) == set(range(start, start + 8))
    assert abs(normalized_kendall_distance(original, indices) - 0.05) <= 0.01


@pytest.mark.parametrize("target", [0.05, 0.10, 0.15, 0.20])
def test_broad_budget_is_deterministic_and_preserves_two_block_orders(target: float) -> None:
    original = np.arange(24)
    first = broad_budget_indices(24, target, 20760)
    second = broad_budget_indices(24, target, 20760)
    assert np.array_equal(first, second)
    assert np.array_equal(np.sort(first), original)
    assert abs(normalized_kendall_distance(original, first) - target) <= 0.01
    drops = np.flatnonzero(np.diff(first) != 1)
    assert len(drops) <= 3
    assert len(drops) >= 2
    assert any(
        np.array_equal(first, np.concatenate((
            np.arange(start),
            np.arange(start + left_length, start + left_length + right_length),
            np.arange(start, start + left_length),
            np.arange(start + left_length + right_length, 24),
        )))
        for start in range(24)
        for left_length in range(2, 24 - start)
        for right_length in range(1, 25 - start - left_length)
    )


def test_eligible_subset_requires_all_seeds_and_both_changed() -> None:
    rows = []
    for user_id in (7, 8, 9):
        for seed in (20760, 20761, 20762):
            rows.append({
                "user_id": user_id, "target_kendall_distance": 0.1,
                "perturbation_seed": seed,
                "local_target_error": 0.005 if user_id != 8 or seed != 20761 else 0.02,
                "broad_target_error": 0.004,
                "pair_kendall_mismatch": 0.009,
                "local_effectively_unchanged": False,
                "broad_effectively_unchanged": user_id == 9 and seed == 20762,
            })
    frame = pd.DataFrame(rows)
    assert structurally_eligible_users(frame, 0.1).tolist() == [7]
