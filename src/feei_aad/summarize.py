from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def load_results(inputs: list[Path]) -> list[dict]:
    rows = []
    seen = set()
    for root in inputs:
        for path in sorted(root.rglob("result.json")):
            row = json.loads(path.read_text(encoding="utf-8"))
            key = (row["view"], row["model"], row["subject"], row["fold"])
            if key in seen:
                raise ValueError(f"Duplicate result for {key}: {path}")
            seen.add(key)
            rows.append(row)
    if not rows:
        raise FileNotFoundError("No result.json files were found")
    return rows


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict], expected_folds: int) -> tuple[list[dict], list[dict]]:
    subject_folds: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        key = row["view"], row["model"], row["subject"]
        subject_folds[key].append(float(row["test_window_accuracy"]))

    participants = []
    for (view, model, subject), values in sorted(subject_folds.items()):
        if len(values) != expected_folds:
            raise ValueError(
                f"{view}/{model}/{subject}: found {len(values)} folds, "
                f"expected {expected_folds}"
            )
        participants.append(
            {
                "view": view,
                "model": model,
                "subject": subject,
                "folds": len(values),
                "accuracy": sum(values) / len(values),
                "accuracy_percent": 100.0 * sum(values) / len(values),
            }
        )

    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in participants:
        grouped[(row["view"], row["model"])].append(float(row["accuracy"]))
    summary = []
    for (view, model), values in sorted(grouped.items()):
        summary.append(
            {
                "view": view,
                "model": model,
                "subjects": len(values),
                "mean_accuracy": sum(values) / len(values),
                "sd_accuracy": statistics.stdev(values),
                "mean_percent": 100.0 * sum(values) / len(values),
                "sd_percent": 100.0 * statistics.stdev(values),
            }
        )
    return participants, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize fold results by subject.")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/generated"))
    parser.add_argument("--expected-folds", type=int, default=5)
    args = parser.parse_args()

    participants, summary = summarize(load_results(args.inputs), args.expected_folds)
    write_csv(
        args.output_dir / "participant_accuracy.csv",
        participants,
        ["view", "model", "subject", "folds", "accuracy", "accuracy_percent"],
    )
    write_csv(
        args.output_dir / "summary_accuracy.csv",
        summary,
        [
            "view",
            "model",
            "subjects",
            "mean_accuracy",
            "sd_accuracy",
            "mean_percent",
            "sd_percent",
        ],
    )
    for row in summary:
        print(
            f"{row['view']:14s} {row['model']:30s} "
            f"{row['mean_percent']:.2f} +/- {row['sd_percent']:.2f}"
        )


if __name__ == "__main__":
    main()
