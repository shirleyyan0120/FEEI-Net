from __future__ import annotations

from pathlib import Path

import numpy as np

from feei_aad.data import WindowedSubject, load_trial_table
from feei_aad.preprocessing import audit_split, fit_csp, prepare_fold, split_indices


ROOT = Path(__file__).resolve().parents[1]


def synthetic_subject() -> WindowedSubject:
    table = load_trial_table(ROOT / "configs" / "trial_groups.csv")
    trial_ids = np.repeat(np.arange(1, 41), 119)
    labels = np.concatenate(
        [np.full(119, table[trial][0], dtype=np.int64) for trial in range(1, 41)]
    )
    groups = np.concatenate(
        [np.full(119, table[trial][1], dtype=np.int64) for trial in range(1, 41)]
    )
    window_ids = np.concatenate(
        [trial * 1000 + np.arange(119) for trial in range(1, 41)]
    )
    return WindowedSubject(
        subject="test",
        view="frontal_ear",
        eeg=np.zeros((4760, 1, 1), dtype=np.float32),
        labels=labels,
        trial_ids=trial_ids,
        pair_groups=groups,
        starts=np.tile(np.arange(119) * 64, 40),
        window_ids=window_ids,
    )


def test_trial_table() -> None:
    rows = load_trial_table(ROOT / "configs" / "trial_groups.csv")
    assert len(rows) == 40
    assert sum(label == 0 for label, _ in rows.values()) == 20
    assert sum(label == 1 for label, _ in rows.values()) == 20


def test_grouped_split_has_no_overlap() -> None:
    data = synthetic_subject()
    for fold_index in range(5):
        indices = split_indices(data, fold_index, split_seed=20260807)
        report = audit_split(data, indices)
        assert report["counts"]["train"] == {
            "windows": 2856,
            "trials": 24,
            "stimulus_pairs": 12,
        }
        assert report["counts"]["validation"] == {
            "windows": 952,
            "trials": 8,
            "stimulus_pairs": 4,
        }
        assert report["counts"]["test"] == {
            "windows": 952,
            "trials": 8,
            "stimulus_pairs": 4,
        }


def test_full_rank_csp() -> None:
    rng = np.random.default_rng(7)
    eeg = rng.normal(size=(24, 8, 128)).astype(np.float32)
    labels = np.repeat([0, 1], 12)
    filters, eigenvalues = fit_csp(eeg, labels)
    assert filters.shape == (8, 8)
    assert eigenvalues.shape == (8,)
    assert np.isfinite(filters).all()


def preprocessing_subject() -> WindowedSubject:
    rng = np.random.default_rng(11)
    windows = 30
    channels = 8
    return WindowedSubject(
        subject="test",
        view="frontal_only",
        eeg=rng.normal(size=(windows, channels, 128)).astype(np.float32),
        labels=np.tile([0, 1], windows // 2),
        trial_ids=np.arange(windows),
        pair_groups=np.arange(windows),
        starts=np.zeros(windows, dtype=np.int64),
        window_ids=np.arange(windows),
    )


def test_dbpnet_hybrid_preprocessing() -> None:
    data = preprocessing_subject()
    indices = (np.arange(20), np.arange(20, 25), np.arange(25, 30))
    blocks, metadata = prepare_fold(data, indices, "dbpnet")
    assert [block.shape for block in blocks] == [(20, 16, 128), (5, 16, 128), (5, 16, 128)]
    assert metadata["model_input_blocks"] == ["anatomical_raw", "csp_components"]
    assert all(np.isfinite(block).all() for block in blocks)


def test_listennet_fold_local_alignment() -> None:
    data = preprocessing_subject()
    indices = (np.arange(20), np.arange(20, 25), np.arange(25, 30))
    blocks, metadata = prepare_fold(data, indices, "listennet")
    assert [block.shape for block in blocks] == [(20, 8, 128), (5, 8, 128), (5, 8, 128)]
    assert metadata["fit_split"] == "inner_train_only"
    assert metadata["application"] == "same_training_fitted_matrix_for_train_validation_test"
    assert all(np.isfinite(block).all() for block in blocks)
