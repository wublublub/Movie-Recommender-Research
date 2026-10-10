import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from movie_recommender.movietweetings_structured_perturbation import (
    block_ranges, perturb_indices, perturb_sequence, perturbation_strength,
)
from movie_recommender.movietweetings_experiment import compact_sets


def test_block_construction_and_partial_final_block() -> None:
    assert block_ranges(11, 4) == [(0, 4), (4, 8), (8, 11)]
    assert block_ranges(3, 4) == [(0, 3)]
    with pytest.raises(ValueError):
        block_ranges(0, 4)


@pytest.mark.parametrize("condition", ["within_block_shuffle", "block_permutation"])
@pytest.mark.parametrize("length", [3, 4, 9, 17])
def test_perturbations_are_deterministic_and_preserve_items(condition: str, length: int) -> None:
    original = np.arange(100, 100 + length, dtype=np.int32)
    before = original.copy()
    first = perturb_sequence(original, 4, condition, 20760)
    second = perturb_sequence(original, 4, condition, 20760)
    assert np.array_equal(first, second)
    assert len(first) == length
    assert np.array_equal(np.sort(first), np.sort(original))
    assert np.array_equal(original, before)
    indices = perturb_indices(length, 4, condition, 20760)
    if condition == "within_block_shuffle":
        for start, end in block_ranges(length, 4):
            assert set(indices[start:end]) == set(range(start, end))
    else:
        for start, end in block_ranges(length, 4):
            original_block = list(range(start, end))
            assert any(
                indices[offset:offset + len(original_block)].tolist() == original_block
                for offset in range(length - len(original_block) + 1)
            )


def test_short_single_block_permutation_is_unchanged() -> None:
    original = np.array([10, 20, 30], dtype=np.int32)
    assert np.array_equal(
        perturb_sequence(original, 4, "block_permutation", 20760), original
    )


def test_strength_calculation_and_target_candidate_invariance() -> None:
    original = np.array([10, 20, 30, 40], dtype=np.int32)
    unchanged = perturbation_strength(original, original)
    assert unchanged["preserved_adjacency_rate"] == 1
    assert unchanged["normalized_displacement"] == 0
    assert unchanged["rank_correlation"] == 1
    assert unchanged["fixed_position_rate"] == 1
    reverse = perturbation_strength(original, original[::-1])
    assert reverse["preserved_adjacency_rate"] == 0
    assert reverse["rank_correlation"] == -1
    for condition in ("within_block_shuffle", "block_permutation"):
        changed = perturb_sequence(original, 2, condition, 20760)
        assert not set(changed) & {50, 60}
    with pytest.raises(ValueError, match="multiset"):
        perturbation_strength(original, np.array([10, 20, 30, 50]))


def test_frozen_targets_and_candidates_are_independent_of_block_order() -> None:
    original = np.array([10, 20, 30, 40], dtype=np.int32)
    full = pd.DataFrame({
        "UserID": [7] * 6, "MovieID": [10, 20, 30, 40, 50, 60],
        "_user_order_rank": list(range(6)),
    })
    split = SimpleNamespace(
        eligible_users=pd.DataFrame({"user_id": [7]}), full_ratings=full,
    )
    targets = full.iloc[4:].copy()
    catalog = np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=np.int64)
    frozen = compact_sets(split, targets, catalog)
    for condition in ("within_block_shuffle", "block_permutation"):
        changed = perturb_sequence(original, 2, condition, 20760)
        assert set(changed) == set(original)
        assert not set(changed) & set(targets["MovieID"])
        repeated = compact_sets(split, targets, catalog)
        assert np.array_equal(repeated.targets[0], frozen.targets[0])
        assert np.array_equal(repeated.excluded[0], frozen.excluded[0])
        assert set(catalog[frozen.targets[0]]) == {50, 60}
        assert set(catalog[frozen.excluded[0]]) == {10, 20, 30, 40}
