from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import platform
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy
import sklearn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .data import VIEWS, WindowedSubject, load_subject_windows
from .models.registry import (
    ABLATION_MODELS,
    BASELINE_MODELS,
    TRAINING_CONFIG,
    build_model,
)
from .models.recent_baselines import (
    RECENT_MODEL_PROVENANCE,
    validate_bilateral_groups,
)
from .preprocessing import N_OUTER_FOLDS, audit_split, prepare_fold, split_indices


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUBJECTS = PACKAGE_ROOT / "configs" / "subjects.txt"
DEFAULT_TRIAL_TABLE = PACKAGE_ROOT / "configs" / "trial_groups.csv"
PROTOCOL_VERSION = "feei-aad-release-v2"


class EEGWindowDataset(Dataset):
    def __init__(self, eeg: np.ndarray, labels: np.ndarray):
        self.eeg = torch.from_numpy(eeg)
        self.labels = torch.from_numpy(labels.astype(np.int64, copy=False))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.eeg[index], self.labels[index]


def read_subjects(path: Path) -> list[str]:
    subjects = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(subjects) != len(set(subjects)):
        raise ValueError(f"Duplicate subject identifiers in {path}")
    return subjects


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device(name: str, cuda_device: int) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        elif torch.backends.mps.is_available():
            name = "mps"
        else:
            name = "cpu"
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.cuda.set_device(cuda_device)
        return torch.device(f"cuda:{cuda_device}")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(name)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, np.ndarray]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    total_loss = 0.0
    with torch.no_grad():
        for eeg, target in loader:
            eeg = eeg.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            logits = model(eeg)
            total_loss += criterion(logits, target).item()
            labels.append(target.cpu().numpy())
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    label_array = np.concatenate(labels)
    probability_array = np.concatenate(probabilities)
    accuracy = float(np.mean(probability_array.argmax(axis=1) == label_array))
    return total_loss / len(label_array), accuracy, probability_array


def write_predictions(
    path: Path,
    data: WindowedSubject,
    test_index: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    predictions = probabilities.argmax(axis=1)
    fields = (
        "window_id",
        "trial_id",
        "stimulus_pair_group",
        "start_sample",
        "label",
        "prediction",
        "prob_0",
        "prob_1",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for local_index, data_index in enumerate(test_index):
            writer.writerow(
                {
                    "window_id": int(data.window_ids[data_index]),
                    "trial_id": int(data.trial_ids[data_index]),
                    "stimulus_pair_group": int(data.pair_groups[data_index]),
                    "start_sample": int(data.starts[data_index]),
                    "label": int(data.labels[data_index]),
                    "prediction": int(predictions[local_index]),
                    "prob_0": float(probabilities[local_index, 0]),
                    "prob_1": float(probabilities[local_index, 1]),
                }
            )


def run_fold(
    data: WindowedSubject,
    model_id: str,
    training_config: dict,
    fold_index: int,
    fold_dir: Path,
    device: torch.device,
    seed: int,
    split_seed: int,
    max_epochs: int,
    patience: int,
    num_workers: int,
    save_checkpoint: bool,
) -> dict:
    result_path = fold_dir / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        expected = {
            "protocol": PROTOCOL_VERSION,
            "model": model_id,
            "subject": data.subject,
            "view": data.view,
            "fold": fold_index + 1,
            "seed": seed,
            "split_seed": split_seed,
        }
        mismatched = {
            key: (result.get(key), value)
            for key, value in expected.items()
            if result.get(key) != value
        }
        if mismatched:
            raise RuntimeError(f"Existing result does not match this run: {mismatched}")
        return result
    fold_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    indices = split_indices(data, fold_index, split_seed)
    split_audit = audit_split(data, indices)
    blocks, normalization = prepare_fold(data, indices, model_id)
    train_index, validation_index, test_index = indices
    labels = (
        data.labels[train_index],
        data.labels[validation_index],
        data.labels[test_index],
    )

    config = training_config
    datasets = [EEGWindowDataset(eeg, target) for eeg, target in zip(blocks, labels)]
    generator = torch.Generator().manual_seed(seed)
    loaders = (
        DataLoader(
            datasets[0],
            batch_size=config["batch_size"],
            shuffle=True,
            generator=generator,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        ),
        DataLoader(
            datasets[1],
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        ),
        DataLoader(
            datasets[2],
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        ),
    )

    model = build_model(
        model_id,
        blocks[0].shape[1],
        blocks[0].shape[2],
        data.view,
    ).to(device)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    waiting = 0
    history = []
    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for eeg, target in loaders[0]:
            eeg = eeg.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(eeg), target)
            loss.backward()
            optimizer.step()
            if hasattr(model, "apply_constraints"):
                model.apply_constraints()
            train_loss += loss.item() * len(target)
            train_count += len(target)

        validation_loss, validation_accuracy, _ = evaluate(model, loaders[1], device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss / train_count,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }
        )
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            waiting = 0
        else:
            waiting += 1
            if waiting >= patience:
                break

    if best_state is None:
        raise RuntimeError("No model checkpoint was selected")
    model.load_state_dict(best_state)
    validation_loss, validation_accuracy, _ = evaluate(model, loaders[1], device)
    test_loss, test_accuracy, test_probabilities = evaluate(model, loaders[2], device)

    if save_checkpoint:
        torch.save(best_state, fold_dir / "best_model.pt")
    write_predictions(
        fold_dir / "predictions.csv",
        data,
        test_index,
        test_probabilities,
    )
    (fold_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    (fold_dir / "normalization.json").write_text(
        json.dumps(normalization, indent=2), encoding="utf-8"
    )
    (fold_dir / "split_audit.json").write_text(
        json.dumps(split_audit, indent=2), encoding="utf-8"
    )
    result = {
        "protocol": PROTOCOL_VERSION,
        "model": model_id,
        "subject": data.subject,
        "view": data.view,
        "input_channels": int(data.eeg.shape[1]),
        "fold": fold_index + 1,
        "seed": seed,
        "split_seed": split_seed,
        "parameters": parameter_count,
        "best_epoch": best_epoch,
        "validation_loss": validation_loss,
        "validation_accuracy": validation_accuracy,
        "test_loss": test_loss,
        "test_window_accuracy": test_accuracy,
        "elapsed_seconds": round(time.time() - start_time, 3),
        "training": dict(config),
        "model_provenance": RECENT_MODEL_PROVENANCE.get(model_id),
        "bilateral_groups": (
            validate_bilateral_groups(data.view, data.eeg.shape[1])
            if model_id == "xanet"
            else None
        ),
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the released 1 s auditory attention decoding experiments."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=PACKAGE_ROOT / "runs")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--models", default=",".join(BASELINE_MODELS))
    parser.add_argument("--views", default=",".join(VIEWS))
    parser.add_argument("--subjects", default="all")
    parser.add_argument("--subjects-file", type=Path, default=DEFAULT_SUBJECTS)
    parser.add_argument("--trial-table", type=Path, default=DEFAULT_TRIAL_TABLE)
    parser.add_argument("--folds", default="1,2,3,4,5")
    parser.add_argument("--base-seed", type=int, default=20261010)
    parser.add_argument("--split-seed", type=int, default=20260807)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--lr", "--learning-rate", dest="learning_rate", type=float, default=None
    )
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--frontal-dir", default="scan")
    parser.add_argument("--ear-dir", default="ear")
    parser.add_argument(
        "--device", choices=("auto", "cuda", "mps", "cpu"), default="auto"
    )
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--no-checkpoint", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    known_models = set(BASELINE_MODELS) | set(ABLATION_MODELS)
    models = parse_list(args.models)
    views = parse_list(args.views)
    if set(models) - known_models:
        raise ValueError(f"Unknown models: {sorted(set(models) - known_models)}")
    if set(views) - set(VIEWS):
        raise ValueError(f"Unknown views: {sorted(set(views) - set(VIEWS))}")
    if set(models) & set(ABLATION_MODELS) and views != ["frontal_ear"]:
        raise ValueError("Progressive ablations require --views frontal_ear")
    if args.learning_rate is not None and args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if args.weight_decay is not None and args.weight_decay < 0:
        raise ValueError("--weight-decay must be non-negative")
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    training_configs = {model: dict(TRAINING_CONFIG[model]) for model in models}
    for config in training_configs.values():
        if args.learning_rate is not None:
            config["lr"] = args.learning_rate
        if args.weight_decay is not None:
            config["weight_decay"] = args.weight_decay
        if args.batch_size is not None:
            config["batch_size"] = args.batch_size

    canonical_subjects = read_subjects(args.subjects_file)
    subjects = (
        canonical_subjects if args.subjects == "all" else parse_list(args.subjects)
    )
    if set(subjects) - set(canonical_subjects):
        raise ValueError("Every requested subject must appear in --subjects-file")
    folds = [int(value) - 1 for value in parse_list(args.folds)]
    if not folds or set(folds) - set(range(N_OUTER_FOLDS)):
        raise ValueError(f"Invalid folds: {args.folds}")

    device = select_device(args.device, args.cuda_device)
    run_dir = args.output_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        metadata = {
            "protocol": PROTOCOL_VERSION,
            "created": datetime.now().isoformat(),
            "models": models,
            "views": views,
            "subjects": subjects,
            "folds": [fold + 1 for fold in folds],
            "base_seed": args.base_seed,
            "split_seed": args.split_seed,
            "primary_endpoint": "1-second test-window accuracy",
            "statistical_unit": "subject",
            "software": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "scikit_learn": sklearn.__version__,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
            },
            "training": training_configs,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    for view in views:
        for subject in subjects:
            data = load_subject_windows(
                args.data_root,
                subject,
                view,
                args.trial_table,
                frontal_dir=args.frontal_dir,
                ear_dir=args.ear_dir,
            )
            subject_position = canonical_subjects.index(subject)
            for model_id in models:
                for fold_index in folds:
                    seed = args.base_seed + 100 * subject_position + fold_index
                    fold_dir = (
                        run_dir / view / model_id / subject / f"fold_{fold_index + 1}"
                    )
                    result = run_fold(
                        data=data,
                        model_id=model_id,
                        training_config=training_configs[model_id],
                        fold_index=fold_index,
                        fold_dir=fold_dir,
                        device=device,
                        seed=seed,
                        split_seed=args.split_seed,
                        max_epochs=args.max_epochs,
                        patience=args.patience,
                        num_workers=args.num_workers,
                        save_checkpoint=not args.no_checkpoint,
                    )
                    print(
                        f"{view}/{subject}/{model_id}/fold{fold_index + 1}: "
                        f"accuracy={result['test_window_accuracy']:.6f}",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
