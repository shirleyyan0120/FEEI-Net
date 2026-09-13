from __future__ import annotations

import numpy as np
import scipy.linalg
from sklearn.model_selection import StratifiedGroupKFold

from .data import WindowedSubject
from .models.registry import (
    CSP_MODELS,
    EUCLIDEAN_ALIGNMENT_MODELS,
    HYBRID_CSP_MODELS,
)


N_OUTER_FOLDS = 5
N_INNER_FOLDS = 4


def split_indices(
    data: WindowedSubject,
    fold_index: int,
    split_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    outer = StratifiedGroupKFold(
        n_splits=N_OUTER_FOLDS,
        shuffle=True,
        random_state=split_seed,
    )
    outer_splits = list(outer.split(data.eeg, data.labels, data.pair_groups))
    outer_train, test_index = outer_splits[fold_index]
    inner = StratifiedGroupKFold(
        n_splits=N_INNER_FOLDS,
        shuffle=True,
        random_state=split_seed + fold_index,
    )
    inner_splits = list(
        inner.split(
            data.eeg[outer_train],
            data.labels[outer_train],
            data.pair_groups[outer_train],
        )
    )
    train_relative, validation_relative = inner_splits[fold_index % N_INNER_FOLDS]
    return outer_train[train_relative], outer_train[validation_relative], test_index


def audit_split(
    data: WindowedSubject,
    indices: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict:
    names = ("train", "validation", "test")
    trial_sets = [set(map(int, np.unique(data.trial_ids[index]))) for index in indices]
    group_sets = [
        set(map(int, np.unique(data.pair_groups[index]))) for index in indices
    ]
    index_sets = [set(map(int, index)) for index in indices]
    window_sets = [set(map(int, data.window_ids[index])) for index in indices]
    pairs = ((0, 1), (0, 2), (1, 2))
    for first, second in pairs:
        if index_sets[first] & index_sets[second]:
            raise RuntimeError("Array indices cross data splits")
        if trial_sets[first] & trial_sets[second]:
            raise RuntimeError("Trials cross data splits")
        if group_sets[first] & group_sets[second]:
            raise RuntimeError("Stimulus pairs cross data splits")
        if window_sets[first] & window_sets[second]:
            raise RuntimeError("Window identifiers cross data splits")
    if set.union(*index_sets) != set(range(len(data.labels))):
        raise RuntimeError("Data splits do not cover every window")
    for name, index in zip(names, indices):
        if set(np.unique(data.labels[index])) != {0, 1}:
            raise RuntimeError(f"{name} split lacks a class")
    return {
        "group_key": "stimulus_pair_group",
        "counts": {
            name: {
                "windows": len(index),
                "trials": len(trial_sets[position]),
                "stimulus_pairs": len(group_sets[position]),
            }
            for position, (name, index) in enumerate(zip(names, indices))
        },
        "trial_ids": {name: sorted(values) for name, values in zip(names, trial_sets)},
        "stimulus_pair_groups": {
            name: sorted(values) for name, values in zip(names, group_sets)
        },
        "overlapping_windows_cross_splits": False,
    }


def _channel_zscore(
    data: WindowedSubject,
    indices: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], dict]:
    train_index = indices[0]
    mean = data.eeg[train_index].mean(axis=(0, 2), keepdims=True)
    std = data.eeg[train_index].std(axis=(0, 2), keepdims=True)
    std[std < 1e-8] = 1.0
    transformed = tuple(
        ((data.eeg[index] - mean) / std).astype(np.float32) for index in indices
    )
    metadata = {
        "fit_split": "inner_train_only",
        "transform": "per_channel_zscore",
        "channel_mean": mean.squeeze().astype(float).tolist(),
        "channel_std": std.squeeze().astype(float).tolist(),
    }
    return transformed, metadata


def fit_csp(eeg: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    covariances = []
    for label in (0, 1):
        class_eeg = eeg[labels == label].astype(np.float64, copy=False)
        samples = class_eeg.transpose(1, 0, 2).reshape(class_eeg.shape[1], -1)
        samples -= samples.mean(axis=1, keepdims=True)
        covariance = samples @ samples.T
        covariance /= max(float(np.trace(covariance)), 1e-12)
        covariances.append(covariance)
    composite = covariances[0] + covariances[1]
    regularization = 1e-6 * np.trace(composite) / composite.shape[0]
    eigenvalues, eigenvectors = scipy.linalg.eigh(
        covariances[0],
        composite + regularization * np.eye(composite.shape[0]),
    )
    order = np.argsort(np.abs(eigenvalues - 0.5))[::-1]
    return (
        eigenvectors[:, order].T.astype(np.float32),
        eigenvalues[order].astype(np.float32),
    )


def _csp_zscore(
    data: WindowedSubject,
    indices: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], dict]:
    train_index = indices[0]
    filters, eigenvalues = fit_csp(data.eeg[train_index], data.labels[train_index])
    components = tuple(
        np.einsum("kc,nct->nkt", filters, data.eeg[index], optimize=True).astype(
            np.float32
        )
        for index in indices
    )
    mean = components[0].mean(axis=(0, 2), keepdims=True)
    std = components[0].std(axis=(0, 2), keepdims=True)
    std[std < 1e-8] = 1.0
    transformed = tuple(
        ((values - mean) / std).astype(np.float32) for values in components
    )
    metadata = {
        "fit_split": "inner_train_only",
        "transform": "full_rank_csp_then_component_zscore",
        "component_mean": mean.squeeze().astype(float).tolist(),
        "component_std": std.squeeze().astype(float).tolist(),
        "ordered_eigenvalues": eigenvalues.astype(float).tolist(),
    }
    return transformed, metadata


def _hybrid_csp_zscore(
    data: WindowedSubject,
    indices: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], dict]:
    raw_blocks, raw_metadata = _channel_zscore(data, indices)
    filters, eigenvalues = fit_csp(raw_blocks[0], data.labels[indices[0]])
    components = tuple(
        np.einsum("kc,nct->nkt", filters, values, optimize=True).astype(np.float32)
        for values in raw_blocks
    )
    mean = components[0].mean(axis=(0, 2), keepdims=True)
    std = components[0].std(axis=(0, 2), keepdims=True)
    std[std < 1e-8] = 1.0
    components = tuple(
        ((values - mean) / std).astype(np.float32) for values in components
    )
    combined = tuple(
        np.concatenate((raw, csp), axis=1).astype(np.float32)
        for raw, csp in zip(raw_blocks, components)
    )
    metadata = {
        "fit_split": "inner_train_only",
        "transform": "raw_channel_zscore_plus_full_rank_csp_component_zscore",
        "raw_channel_mean": raw_metadata["channel_mean"],
        "raw_channel_std": raw_metadata["channel_std"],
        "component_mean": mean.squeeze().astype(float).tolist(),
        "component_std": std.squeeze().astype(float).tolist(),
        "ordered_eigenvalues": eigenvalues.astype(float).tolist(),
        "model_input_blocks": ["anatomical_raw", "csp_components"],
    }
    return combined, metadata


def _euclidean_alignment(
    data: WindowedSubject,
    indices: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], dict]:
    normalized, zscore_metadata = _channel_zscore(data, indices)
    train = normalized[0].astype(np.float64, copy=False)
    covariance = np.einsum("nct,ndt->cd", train, train, optimize=True)
    covariance /= max(float(train.shape[0] * train.shape[2]), 1.0)
    eigenvalues, eigenvectors = scipy.linalg.eigh(covariance)
    floor = max(float(eigenvalues.max()), 1e-12) * 1e-6
    stabilized = np.maximum(eigenvalues, floor)
    alignment = eigenvectors @ np.diag(stabilized ** -0.5) @ eigenvectors.T
    transformed = tuple(
        np.einsum("cd,ndt->nct", alignment, values, optimize=True).astype(
            np.float32
        )
        for values in normalized
    )
    metadata = {
        "fit_split": "inner_train_only",
        "transform": "train_fitted_channel_zscore_then_euclidean_alignment",
        "channel_mean": zscore_metadata["channel_mean"],
        "channel_std": zscore_metadata["channel_std"],
        "ea_covariance_eigenvalues": eigenvalues.astype(float).tolist(),
        "ea_eigenvalue_floor": float(floor),
        "ea_matrix": alignment.astype(float).tolist(),
        "application": "same_training_fitted_matrix_for_train_validation_test",
    }
    return transformed, metadata


def prepare_fold(
    data: WindowedSubject,
    indices: tuple[np.ndarray, np.ndarray, np.ndarray],
    model_id: str,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], dict]:
    if model_id in HYBRID_CSP_MODELS:
        return _hybrid_csp_zscore(data, indices)
    if model_id in EUCLIDEAN_ALIGNMENT_MODELS:
        return _euclidean_alignment(data, indices)
    if model_id in CSP_MODELS:
        return _csp_zscore(data, indices)
    return _channel_zscore(data, indices)
