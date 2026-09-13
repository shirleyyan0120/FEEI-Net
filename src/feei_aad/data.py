from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.signal import butter, sosfiltfilt


SAMPLING_RATE = 128
BANDPASS = (1.0, 32.0)
WINDOW_SECONDS = 1.0
STEP_SECONDS = 0.5
VIEWS = ("frontal_only", "ear_only", "frontal_ear")


@dataclass(frozen=True)
class WindowedSubject:
    subject: str
    view: str
    eeg: np.ndarray
    labels: np.ndarray
    trial_ids: np.ndarray
    pair_groups: np.ndarray
    starts: np.ndarray
    window_ids: np.ndarray
    sampling_rate: int = SAMPLING_RATE


def load_trial_table(path: Path) -> dict[int, tuple[int, int]]:
    rows: dict[int, tuple[int, int]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[int(row["trial"])] = (int(row["label"]), int(row["pair_group"]))

    if set(rows) != set(range(1, 41)):
        raise ValueError("The trial table must contain trials 1 through 40")
    groups = np.asarray([rows[trial][1] for trial in range(1, 41)])
    labels = np.asarray([rows[trial][0] for trial in range(1, 41)])
    unique, counts = np.unique(groups, return_counts=True)
    if len(unique) != 20 or set(counts) != {2}:
        raise ValueError("Expected 20 stimulus-pair groups with two trials each")
    for group in unique:
        if len(np.unique(labels[groups == group])) != 1:
            raise ValueError(f"Stimulus-pair group {group} crosses labels")
    return rows


def _load_eeg(path: Path, channels: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    keys = [key for key in payload if key.startswith("EEG_")]
    if len(keys) != 1:
        raise ValueError(f"{path}: expected one EEG_* variable, found {keys}")
    values = np.asarray(payload[keys[0]].data, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < channels:
        raise ValueError(f"{path}: invalid EEG shape {values.shape}")
    values = values[:channels]
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: non-finite samples")
    return values


def _read_trial(
    data_root: Path,
    subject: str,
    trial: int,
    view: str,
    frontal_dir: str,
    ear_dir: str,
) -> np.ndarray:
    frontal_path = data_root / subject / frontal_dir / "128" / f"{trial}.mat"
    ear_path = data_root / subject / ear_dir / "128" / f"{trial}.mat"
    if view == "frontal_only":
        return _load_eeg(frontal_path, 8)
    if view == "ear_only":
        return _load_eeg(ear_path, 20)
    if view == "frontal_ear":
        frontal = _load_eeg(frontal_path, 8)
        ear = _load_eeg(ear_path, 20)
        if frontal.shape[1] != ear.shape[1]:
            raise ValueError(f"{subject}/{trial}: device sample counts differ")
        return np.concatenate([frontal, ear], axis=0)
    raise ValueError(f"Unknown view: {view}")


def _bandpass(values: np.ndarray) -> np.ndarray:
    sos = butter(8, BANDPASS, btype="bandpass", fs=SAMPLING_RATE, output="sos")
    return sosfiltfilt(sos, values, axis=-1)


def load_subject_windows(
    data_root: Path,
    subject: str,
    view: str,
    trial_table_path: Path,
    frontal_dir: str = "scan",
    ear_dir: str = "ear",
) -> WindowedSubject:
    if view not in VIEWS:
        raise ValueError(f"Unknown view: {view}")
    trial_table = load_trial_table(trial_table_path)
    window_samples = round(WINDOW_SECONDS * SAMPLING_RATE)
    step_samples = round(STEP_SECONDS * SAMPLING_RATE)

    windows: list[np.ndarray] = []
    labels: list[int] = []
    trial_ids: list[int] = []
    pair_groups: list[int] = []
    starts: list[int] = []
    window_ids: list[int] = []

    for trial in range(1, 41):
        raw = _read_trial(
            data_root,
            subject,
            trial,
            view,
            frontal_dir,
            ear_dir,
        )
        if raw.shape[1] != 60 * SAMPLING_RATE:
            raise ValueError(
                f"{subject}/{trial}: expected a 60 s trial, got {raw.shape}"
            )
        filtered = _bandpass(raw)
        label, pair_group = trial_table[trial]
        for window_index, start in enumerate(
            range(0, filtered.shape[1] - window_samples + 1, step_samples)
        ):
            windows.append(
                filtered[:, start : start + window_samples].astype(np.float32)
            )
            labels.append(label)
            trial_ids.append(trial)
            pair_groups.append(pair_group)
            starts.append(start)
            window_ids.append(trial * 1000 + window_index)

    expected_channels = {"frontal_only": 8, "ear_only": 20, "frontal_ear": 28}[view]
    eeg = np.stack(windows)
    if eeg.shape != (4760, expected_channels, 128):
        raise ValueError(f"Unexpected window tensor {eeg.shape}")
    return WindowedSubject(
        subject=subject,
        view=view,
        eeg=eeg,
        labels=np.asarray(labels, dtype=np.int64),
        trial_ids=np.asarray(trial_ids, dtype=np.int64),
        pair_groups=np.asarray(pair_groups, dtype=np.int64),
        starts=np.asarray(starts, dtype=np.int64),
        window_ids=np.asarray(window_ids, dtype=np.int64),
    )
