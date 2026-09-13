from __future__ import annotations

import pytest
import torch

from feei_aad.models.registry import (
    ABLATION_MODELS,
    BASELINE_MODELS,
    TRAINING_CONFIG,
    build_model,
)
from feei_aad.models.recent_baselines import validate_bilateral_groups


VIEW_CHANNELS = {
    "frontal_only": 8,
    "ear_only": 20,
    "frontal_ear": 28,
}


@pytest.mark.parametrize("model_id", BASELINE_MODELS)
@pytest.mark.parametrize("view,channels", VIEW_CHANNELS.items())
def test_baseline_forward(model_id: str, view: str, channels: int) -> None:
    model_channels = 2 * channels if model_id == "dbpnet" else channels
    model = build_model(model_id, model_channels, 128, view).eval()
    with torch.no_grad():
        logits = model(torch.randn(2, model_channels, 128))
    assert logits.shape == (2, 2)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("model_id", ABLATION_MODELS)
def test_ablation_forward(model_id: str) -> None:
    model = build_model(model_id, 28, 128, "frontal_ear").eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 28, 128))
    assert logits.shape == (2, 2)
    assert torch.isfinite(logits).all()


def test_parameter_matched_carrier_levels() -> None:
    counts = {
        model_id: sum(
            parameter.numel()
            for parameter in build_model(
                model_id, 28, 128, "frontal_ear"
            ).parameters()
        )
        for model_id in (
            "dual_branch",
            "dual_branch_cd",
            "dual_branch_cd_interaction",
        )
    }
    assert len(set(counts.values())) == 1


def test_unified_parameter_count() -> None:
    model = build_model("unified_carrier", 28, 128, "frontal_ear")
    assert sum(parameter.numel() for parameter in model.parameters()) == 243462


def test_shared_training_configuration() -> None:
    expected = {"lr": 3e-4, "weight_decay": 3e-4, "batch_size": 32}
    assert set(TRAINING_CONFIG) == set(BASELINE_MODELS) | set(ABLATION_MODELS)
    assert all(config == expected for config in TRAINING_CONFIG.values())


def test_public_baseline_index() -> None:
    assert BASELINE_MODELS == (
        "stanet",
        "xanet",
        "darnet",
        "dbpnet",
        "listennet",
        "mhanet",
        "hcan",
    )


@pytest.mark.parametrize("view,channels", VIEW_CHANNELS.items())
def test_xanet_bilateral_groups(view: str, channels: int) -> None:
    groups = validate_bilateral_groups(view, channels)
    right = set(groups["participant_right"])
    left = set(groups["participant_left"])
    assert not right & left
    assert right | left == set(range(channels))


def test_recent_baseline_parameter_counts() -> None:
    expected = {
        "frontal_only": {"dbpnet": 865202, "listennet": 2334, "xanet": 93138},
        "ear_only": {"dbpnet": 872738, "listennet": 3054, "xanet": 93906},
        "frontal_ear": {"dbpnet": 878722, "listennet": 3438, "xanet": 94418},
    }
    for view, channels in VIEW_CHANNELS.items():
        for model_id, parameter_count in expected[view].items():
            model_channels = 2 * channels if model_id == "dbpnet" else channels
            model = build_model(model_id, model_channels, 128, view)
            assert sum(parameter.numel() for parameter in model.parameters()) == parameter_count
