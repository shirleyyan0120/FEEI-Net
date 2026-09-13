from __future__ import annotations

import torch.nn as nn

from .baselines import DARNet, HCAN, MHANet, STAnet
from .feei import (
    FAIE096BranchInteractionNet,
    FAIE121SpatialInteractionNet,
    FAIE126UnifiedCarrierNet,
    FAIE130DualBranchNoCDNet,
)
from .recent_baselines import RECENT_MODEL_PROVENANCE, build_recent_baseline


BASELINE_MODELS = (
    "stanet",
    "xanet",
    "darnet",
    "dbpnet",
    "listennet",
    "mhanet",
    "hcan",
)

ABLATION_MODELS = (
    "unified_carrier",
    "dual_branch",
    "dual_branch_cd",
    "dual_branch_cd_interaction",
    "feei_full",
)

CSP_MODELS = frozenset({"darnet", "mhanet", "hcan"})
HYBRID_CSP_MODELS = frozenset({"dbpnet"})
EUCLIDEAN_ALIGNMENT_MODELS = frozenset({"listennet"})

TRAINING_CONFIG = {
    "stanet": {"lr": 3e-4, "weight_decay": 3e-4, "batch_size": 32},
    "xanet": {"lr": 3e-4, "weight_decay": 3e-4, "batch_size": 32},
    "darnet": {"lr": 3e-4, "weight_decay": 3e-4, "batch_size": 32},
    "dbpnet": {"lr": 3e-4, "weight_decay": 3e-4, "batch_size": 32},
    "listennet": {"lr": 3e-4, "weight_decay": 3e-4, "batch_size": 32},
    "mhanet": {"lr": 3e-4, "weight_decay": 3e-4, "batch_size": 32},
    "hcan": {"lr": 3e-4, "weight_decay": 3e-4, "batch_size": 32},
    "unified_carrier": {"lr": 3e-4, "weight_decay": 3e-4, "batch_size": 32},
    "dual_branch": {"lr": 3e-4, "weight_decay": 3e-4, "batch_size": 32},
    "dual_branch_cd": {"lr": 3e-4, "weight_decay": 3e-4, "batch_size": 32},
    "dual_branch_cd_interaction": {
        "lr": 3e-4,
        "weight_decay": 3e-4,
        "batch_size": 32,
    },
    "feei_full": {"lr": 3e-4, "weight_decay": 3e-4, "batch_size": 32},
}


def build_model(
    model_id: str,
    num_channels: int,
    time_samples: int,
    channel_view: str,
) -> nn.Module:
    if model_id == "stanet":
        return STAnet(num_channels)
    if model_id == "darnet":
        return DARNet(num_channels)
    if model_id == "mhanet":
        return MHANet(num_channels, time_samples)
    if model_id == "hcan":
        return HCAN(num_channels)
    if model_id in RECENT_MODEL_PROVENANCE:
        return build_recent_baseline(
            model_id,
            num_channels,
            time_samples,
            channel_view,
        )
    if model_id == "unified_carrier":
        return FAIE126UnifiedCarrierNet(
            num_channels,
            time_samples,
            variant="fai_e126_unified_28ch_carrier",
        )
    if model_id == "dual_branch":
        return FAIE130DualBranchNoCDNet(
            num_channels,
            time_samples,
            variant="fai_e130_dual_branch_no_cd_none",
        )
    if model_id == "dual_branch_cd":
        return FAIE096BranchInteractionNet(
            num_channels,
            time_samples,
            variant="fai_e096_branch_interaction_none",
        )
    if model_id == "dual_branch_cd_interaction":
        return FAIE096BranchInteractionNet(
            num_channels,
            time_samples,
            variant="fai_e096_branch_interaction_full",
        )
    if model_id == "feei_full":
        return FAIE121SpatialInteractionNet(
            num_channels,
            time_samples,
            variant="fai_e121_spatial_full",
        )
    raise ValueError(f"Unknown model: {model_id}")
