import numpy as np
import pytest

from movie_recommender.movietweetings_matched_targeted_experiment import (
    expected_new_cells, hierarchical_paired_intervals,
)


def test_frozen_targeted_grid_contains_exactly_ninety_distinct_cells() -> None:
    cells = expected_new_cells()
    assert len(cells) == 90
    assert len(set(cells)) == 90
    assert {cell[0] for cell in cells} == {"fpmc", "lstm", "sasrec"}
    assert {cell[1] for cell in cells} == {760, 761, 762, 763, 764}
    assert {cell[2] for cell in cells} == {20760, 20761, 20762}
    assert {cell[3] for cell in cells} == {"matched_local", "matched_broad"}


def test_hierarchical_bootstrap_keeps_paired_effects_and_is_deterministic() -> None:
    effects = np.empty((2, 8, 5, 3), dtype=float)
    base = np.arange(8, dtype=float)[:, None, None] / 100
    effects[0] = base + np.arange(5)[None, :, None] / 1000
    effects[1] = effects[0] + 0.25
    first_lower, first_upper = hierarchical_paired_intervals(effects, samples=100, seed=760)
    second_lower, second_upper = hierarchical_paired_intervals(effects, samples=100, seed=760)
    assert first_lower == pytest.approx(second_lower)
    assert first_upper == pytest.approx(second_upper)
    assert first_lower[1] - first_lower[0] == pytest.approx(0.25)
    assert first_upper[1] - first_upper[0] == pytest.approx(0.25)
