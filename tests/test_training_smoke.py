from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from feei_aad.data import WindowedSubject, load_trial_table
from feei_aad.train import run_fold


ROOT = Path(__file__).resolve().parents[1]


def training_subject() -> WindowedSubject:
    table = load_trial_table(ROOT / "configs" / "trial_groups.csv")
    rng = np.random.default_rng(23)
    windows_per_trial = 2
    trial_ids = np.repeat(np.arange(1, 41), windows_per_trial)
    labels = np.concatenate(
        [
            np.full(windows_per_trial, table[trial][0], dtype=np.int64)
            for trial in range(1, 41)
        ]
    )
    groups = np.concatenate(
        [
            np.full(windows_per_trial, table[trial][1], dtype=np.int64)
            for trial in range(1, 41)
        ]
    )
    return WindowedSubject(
        subject="test",
        view="frontal_ear",
        eeg=rng.normal(size=(len(labels), 28, 128)).astype(np.float32),
        labels=labels,
        trial_ids=trial_ids,
        pair_groups=groups,
        starts=np.tile([0, 64], 40),
        window_ids=np.arange(len(labels)),
    )


@pytest.mark.parametrize("model_id", ["dbpnet", "listennet", "xanet"])
def test_recent_baseline_training_step(tmp_path: Path, model_id: str) -> None:
    result = run_fold(
        data=training_subject(),
        model_id=model_id,
        training_config={"lr": 3e-4, "weight_decay": 3e-4, "batch_size": 32},
        fold_index=0,
        fold_dir=tmp_path / model_id,
        device=torch.device("cpu"),
        seed=20261010,
        split_seed=20260807,
        max_epochs=1,
        patience=1,
        num_workers=0,
        save_checkpoint=False,
    )
    assert 0.0 <= result["test_window_accuracy"] <= 1.0
    assert result["training"] == {
        "lr": 3e-4,
        "weight_decay": 3e-4,
        "batch_size": 32,
    }
    assert (tmp_path / model_id / "predictions.csv").exists()
    assert (tmp_path / model_id / "normalization.json").exists()
