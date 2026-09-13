from __future__ import annotations

import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def test_release_text_has_no_chinese_characters() -> None:
    suffixes = {".py", ".md", ".sh", ".sbatch", ".toml", ".txt", ".csv"}
    for path in ROOT.rglob("*"):
        if path.is_file() and path.suffix in suffixes:
            text = path.read_text(encoding="utf-8")
            assert CJK.search(text) is None, path


def read_participant_groups(path: Path) -> dict[tuple[str, str], list[float]]:
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            groups[(row["view"], row["model"])].append(float(row["accuracy"]))
    return groups


def test_reported_summaries_match_participant_values() -> None:
    baseline_groups = read_participant_groups(
        ROOT / "results" / "baseline_participant_accuracy_seed20261010.csv"
    )
    with (ROOT / "results" / "baselines_seed20261010.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        for row in csv.DictReader(handle):
            for view, prefix in (
                ("frontal_only", "frontal_only"),
                ("ear_only", "ear_only"),
                ("frontal_ear", "frontal_ear"),
            ):
                values = baseline_groups[(view, row["model"])]
                assert len(values) == 15
                assert 100 * statistics.mean(values) == pytest.approx(
                    float(row[f"{prefix}_mean_percent"]), abs=1e-6
                )
                assert 100 * statistics.stdev(values) == pytest.approx(
                    float(row[f"{prefix}_sd_percent"]), abs=1e-6
                )

    ablation_groups = read_participant_groups(
        ROOT / "results" / "ablation_participant_accuracy_seed20261010.csv"
    )
    with (ROOT / "results" / "progressive_ablation_seed20261010.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        for row in csv.DictReader(handle):
            values = ablation_groups[("frontal_ear", row["model"])]
            assert len(values) == 15
            assert 100 * statistics.mean(values) == pytest.approx(
                float(row["mean_percent"]), abs=1e-6
            )
            assert 100 * statistics.stdev(values) == pytest.approx(
                float(row["sd_percent"]), abs=1e-6
            )
