from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Legacy scan and ocular names are kept for checkpoint compatibility.
# They denote the eight frontal sensors, not a physiological source.


class FAIE096BranchInteractionNet(nn.Module):
    VALID_VARIANTS = {
        "fai_e096_branch_interaction_full",
        "fai_e096_branch_interaction_none",
    }
    BAND_NAMES = ("delta", "theta", "alpha", "beta")
    BAND_LIMITS = ((1.0, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0))

    def __init__(self, num_channels: int, time_samples: int, variant: str):
        super().__init__()
        if num_channels != 28:
            raise ValueError("E096 scan+ear screen supports exactly 28 channels")
        if variant not in FAIE096BranchInteractionNet.VALID_VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        self.num_channels = int(num_channels)
        self.time_samples = int(time_samples)
        self.feature_maps = 32
        self.branch_maps = 8
        self.time_bins = 4
        self.scan_channel_indices = tuple(range(8))
        self.ear_channel_indices = tuple(range(8, 28))
        self.scan_channels = 8
        self.ear_channels = 20
        self.group_definition = (
            "scan_1_4_right_5_8_left_plus_cEEgrid_1_10_right_11_20_left"
        )
        frequencies = torch.fft.rfftfreq(self.time_samples, d=1.0 / 128.0)
        self.register_buffer(
            "band_masks",
            torch.stack(
                [
                    ((frequencies >= low) & (frequencies < high)).float()
                    for low, high in self.BAND_LIMITS
                ]
            ),
        )
        self.band_logits = nn.Parameter(torch.zeros(len(self.BAND_NAMES)))
        self.scan_temporal_branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        1,
                        self.branch_maps,
                        kernel_size=(1, kernel),
                        padding=(0, kernel // 2),
                        bias=False,
                    ),
                    nn.GroupNorm(4, self.branch_maps),
                    nn.GELU(),
                )
                for kernel in (3, 7, 15)
            ]
        )
        self.scan_side_projection = nn.Sequential(
            nn.Conv2d(2, 3 * self.branch_maps, kernel_size=1, bias=False),
            nn.GroupNorm(4, 3 * self.branch_maps),
            nn.GELU(),
        )
        self.scan_projection = nn.Sequential(
            nn.Conv2d(
                3 * self.branch_maps, self.feature_maps, kernel_size=1, bias=False
            ),
            nn.GroupNorm(8, self.feature_maps),
            nn.GELU(),
            nn.Conv2d(
                self.feature_maps,
                self.feature_maps,
                kernel_size=(3, 5),
                padding=(1, 2),
                bias=False,
            ),
            nn.GroupNorm(8, self.feature_maps),
            nn.GELU(),
        )
        self.ear_temporal_branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        1,
                        self.branch_maps,
                        kernel_size=(1, kernel),
                        padding=(0, kernel // 2),
                        bias=False,
                    ),
                    nn.GroupNorm(4, self.branch_maps),
                    nn.GELU(),
                )
                for kernel in (5, 11, 23)
            ]
        )
        self.ear_projection = nn.Sequential(
            nn.Conv2d(
                3 * self.branch_maps, self.feature_maps, kernel_size=1, bias=False
            ),
            nn.GroupNorm(8, self.feature_maps),
            nn.GELU(),
            nn.Conv2d(
                self.feature_maps,
                self.feature_maps,
                kernel_size=(3, 7),
                padding=(1, 3),
                bias=False,
            ),
            nn.GroupNorm(8, self.feature_maps),
            nn.GELU(),
        )
        self.scan_to_ear = self._make_route()
        self.ear_to_scan = self._make_route()
        self.scan_to_ear_scale_logit = nn.Parameter(torch.tensor(-1.3862944))
        self.ear_to_scan_scale_logit = nn.Parameter(torch.tensor(-1.3862944))
        self.band_branch_fusion = nn.Sequential(
            nn.Conv2d(self.feature_maps, 48, kernel_size=1, bias=False),
            nn.GroupNorm(8, 48),
            nn.GELU(),
        )
        self.band_fusion = nn.Sequential(
            nn.Conv2d(4 * 48, 64, kernel_size=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        self.spatial_collapse = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(num_channels, 1), bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        self.temporal_refine = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=9, padding=4, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(64, 2),
        )
        self._collect_window_diagnostics = False
        self._window_diagnostic_batches: list[dict[str, torch.Tensor]] = []

    def _make_route(self) -> nn.ModuleDict:
        return nn.ModuleDict(
            {
                "query": nn.Linear(self.feature_maps, self.feature_maps, bias=False),
                "key": nn.Linear(self.feature_maps, self.feature_maps, bias=False),
                "value": nn.Linear(self.feature_maps, self.feature_maps, bias=False),
                "out": nn.Sequential(
                    nn.Linear(2 * self.feature_maps, self.feature_maps, bias=False),
                    nn.LayerNorm(self.feature_maps),
                    nn.GELU(),
                ),
                "gate": nn.Sequential(
                    nn.Linear(3 * self.feature_maps, self.feature_maps // 2),
                    nn.GELU(),
                    nn.Linear(self.feature_maps // 2, 1),
                ),
            }
        )

    def _tokens(self, features: torch.Tensor, channels: int) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(features, (channels, self.time_bins))
        return pooled.permute(0, 2, 3, 1).reshape(
            features.shape[0], channels * self.time_bins, self.feature_maps
        )

    def _route(
        self,
        target: torch.Tensor,
        source: torch.Tensor,
        route: nn.ModuleDict,
        scale_parameter: torch.Tensor,
    ):
        query = route["query"](target)
        key = route["key"](source)
        value = route["value"](source)
        weights = torch.softmax(
            torch.matmul(query, key.transpose(1, 2)) / math.sqrt(self.feature_maps),
            dim=-1,
        )
        message = torch.matmul(weights, value)
        residual = route["out"](torch.cat([target, message], dim=-1))
        target_context = target.mean(dim=1)
        source_context = source.mean(dim=1)
        gate = torch.sigmoid(
            route["gate"](
                torch.cat(
                    [target_context, source_context, target_context * source_context],
                    dim=-1,
                )
            )
        ).squeeze(-1)
        residual = 0.5 * torch.sigmoid(scale_parameter) * gate[:, None, None] * residual
        entropy = -(weights.clamp_min(1e-08) * weights.clamp_min(1e-08).log()).sum(
            dim=-1
        )
        entropy = entropy / math.log(max(source.shape[1], 2))
        consistency = (
            F.cosine_similarity(target_context, source_context, dim=-1) + 1.0
        ) * 0.5
        return (residual, gate, entropy.mean(dim=1), consistency)

    @staticmethod
    def _scatter_tokens(
        residual: torch.Tensor,
        channels: int,
        target_time: int,
        channel_start: int,
        total_channels: int,
    ) -> torch.Tensor:
        batch, _, width = residual.shape
        values = residual.reshape(batch, channels, 4, width).permute(0, 3, 1, 2)
        values = values.reshape(batch * width * channels, 1, 4)
        values = F.interpolate(
            values, size=target_time, mode="linear", align_corners=False
        )
        values = values.reshape(batch, width, channels, target_time)
        output = residual.new_zeros(batch, width, total_channels, target_time)
        output[:, :, channel_start : channel_start + channels, :] = values
        return output

    def _scan_encode(self, signal: torch.Tensor) -> torch.Tensor:
        branches = [
            branch(signal.unsqueeze(1)) for branch in self.scan_temporal_branches
        ]
        right = signal[:, :4].mean(dim=1, keepdim=True)
        left = signal[:, 4:8].mean(dim=1, keepdim=True)
        side = self.scan_side_projection(
            torch.cat([0.5 * (right + left), 0.5 * (right - left)], dim=1).unsqueeze(2)
        ).expand(-1, -1, 8, -1)
        return self.scan_projection(torch.cat(branches, dim=1) + side)

    def _ear_encode(self, signal: torch.Tensor) -> torch.Tensor:
        return self.ear_projection(
            torch.cat(
                [branch(signal.unsqueeze(1)) for branch in self.ear_temporal_branches],
                dim=1,
            )
        )

    def set_diagnostic_collection(self, enabled: bool) -> None:
        self._collect_window_diagnostics = bool(enabled)
        if enabled:
            self._window_diagnostic_batches = []

    def _record_diagnostics(self, values: dict[str, torch.Tensor]) -> None:
        if self._collect_window_diagnostics:
            self._window_diagnostic_batches.append(
                {key: value.detach().cpu() for key, value in values.items()}
            )

    def consume_window_diagnostics(self) -> dict[str, list[float]]:
        if not self._window_diagnostic_batches:
            return {}
        keys = self._window_diagnostic_batches[0]
        payload = {
            key: torch.cat([batch[key] for batch in self._window_diagnostic_batches])
            .numpy()
            .astype(float)
            .tolist()
            for key in keys
        }
        self._window_diagnostic_batches = []
        return payload

    def diagnostics(self) -> dict:
        weights = torch.softmax(self.band_logits, dim=0).detach().cpu().tolist()
        return {
            "model_family": "fronto_auricular_branch_backbone_interaction",
            "variant": self.variant,
            "bands": {
                name: list(limits)
                for name, limits in zip(self.BAND_NAMES, self.BAND_LIMITS)
            },
            "branch_design": "dedicated_scan_ocular_and_ear_temporal_encoders",
            "scan_feature": "raw_scan_plus_label_free_common_lateral_difference",
            "interaction_injection": "branch_token_exchange_before_band_fusion",
            "band_weights": dict(zip(self.BAND_NAMES, weights)),
            "group_definition": self.group_definition,
            "frontal_channels": self.scan_channels,
            "ear_channels": self.ear_channels,
            "input_window_samples": self.time_samples,
            "uses_trial_history": False,
            "uses_attention_label_as_input": False,
            "uses_speech_or_envelopes": False,
            "uses_probability_fusion": False,
            "none_only_disables_interaction_residual": True,
        }

    def forward(
        self, eeg: torch.Tensor, envelopes: torch.Tensor | None = None
    ) -> torch.Tensor:
        del envelopes
        batch = eeg.shape[0]
        spectrum = torch.fft.rfft(eeg, n=self.time_samples, dim=-1)
        bands = torch.fft.irfft(
            spectrum[:, None, :, :] * self.band_masks[None, :, None, :],
            n=self.time_samples,
            dim=-1,
        )
        weights = torch.softmax(self.band_logits, dim=0)
        band_maps = []
        diagnostic_batches = []
        for index, name in enumerate(self.BAND_NAMES):
            scan = self._scan_encode(bands[:, index, :8])
            ear = self._ear_encode(bands[:, index, 8:])
            scan_tokens = self._tokens(scan, self.scan_channels)
            ear_tokens = self._tokens(ear, self.ear_channels)
            s2e, s2e_gate, s2e_entropy, s2e_consistency = self._route(
                ear_tokens, scan_tokens, self.scan_to_ear, self.scan_to_ear_scale_logit
            )
            e2s, e2s_gate, e2s_entropy, e2s_consistency = self._route(
                scan_tokens, ear_tokens, self.ear_to_scan, self.ear_to_scan_scale_logit
            )
            if self.variant.endswith("_none"):
                s2e = s2e * 0.0
                e2s = e2s * 0.0
            scan_residual = self._scatter_tokens(
                e2s, self.scan_channels, scan.shape[-1], 0, self.scan_channels
            )
            ear_residual = self._scatter_tokens(
                s2e, self.ear_channels, ear.shape[-1], 0, self.ear_channels
            )
            branch_map = torch.cat([scan + scan_residual, ear + ear_residual], dim=2)
            branch_map = self.band_branch_fusion(branch_map)
            base_rms = branch_map.square().mean(dim=(1, 2, 3)).sqrt().clamp_min(1e-06)
            residual = torch.cat([scan_residual, ear_residual], dim=2)
            diagnostic_batches.append(
                {
                    "scan_to_ear_gate": s2e_gate,
                    "ear_to_scan_gate": e2s_gate,
                    "scan_to_ear_residual_ratio": s2e.square().mean(dim=(1, 2)).sqrt()
                    / base_rms,
                    "ear_to_scan_residual_ratio": e2s.square().mean(dim=(1, 2)).sqrt()
                    / base_rms,
                    "bidirectional_residual_ratio": residual.square()
                    .mean(dim=(1, 2, 3))
                    .sqrt()
                    / base_rms,
                    "scan_to_ear_attention_entropy": s2e_entropy,
                    "ear_to_scan_attention_entropy": e2s_entropy,
                    "connection_consistency": (s2e_consistency + e2s_consistency) * 0.5,
                    "band_weight": weights[index].expand(batch),
                }
            )
            band_maps.append(branch_map * weights[index])
        fused = self.band_fusion(torch.cat(band_maps, dim=1))
        values = self.spatial_collapse(fused).squeeze(2)
        values = self.temporal_refine(values)
        pooled = torch.cat(
            [values.mean(dim=-1), values.std(dim=-1, unbiased=False)], dim=1
        )
        if diagnostic_batches:
            diagnostics = {
                key: torch.stack(
                    [item[key] for item in diagnostic_batches], dim=1
                ).mean(dim=1)
                for key in diagnostic_batches[0]
            }
            diagnostics["band_fusion_entropy"] = (
                -(weights * weights.clamp_min(1e-08).log()).sum() / math.log(4.0)
            ).expand(batch)
            self._record_diagnostics(diagnostics)
        return self.classifier(pooled)


class FAIE102SpatialResidualNet(FAIE096BranchInteractionNet):
    VALID_VARIANTS = {
        "fai_e102_spatial_residual_full",
        "fai_e102_spatial_residual_none",
    }

    def __init__(self, num_channels: int, time_samples: int, variant: str):
        if variant not in FAIE102SpatialResidualNet.VALID_VARIANTS:
            raise ValueError(variant)
        super().__init__(num_channels, time_samples, "fai_e096_branch_interaction_full")
        self.variant = variant
        self.spatial_maps = 64
        self.time_bins = 4
        relation_types = torch.empty((28, 28), dtype=torch.long)
        for target in range(28):
            for source in range(28):
                if target == source:
                    relation_types[target, source] = 0
                elif target < 8 and source < 8:
                    relation_types[target, source] = (
                        1 if target // 4 == source // 4 else 2
                    )
                elif target >= 8 and source >= 8:
                    relation_types[target, source] = (
                        3 if (target - 8) // 10 == (source - 8) // 10 else 4
                    )
                else:
                    relation_types[target, source] = 5
        self.register_buffer("relation_types", relation_types, persistent=False)
        if variant.endswith("_full"):
            self.spatial_q = nn.Linear(self.spatial_maps, self.spatial_maps, bias=False)
            self.spatial_k = nn.Linear(self.spatial_maps, self.spatial_maps, bias=False)
            self.spatial_v = nn.Linear(self.spatial_maps, self.spatial_maps, bias=False)
            self.spatial_out = nn.Sequential(
                nn.Linear(2 * self.spatial_maps, self.spatial_maps, bias=False),
                nn.LayerNorm(self.spatial_maps),
                nn.GELU(),
            )
            self.spatial_gate = nn.Sequential(
                nn.Linear(self.spatial_maps, self.spatial_maps // 2),
                nn.GELU(),
                nn.Linear(self.spatial_maps // 2, 1),
            )
            nn.init.zeros_(self.spatial_gate[-1].weight)
            nn.init.zeros_(self.spatial_gate[-1].bias)
            self.spatial_relation_bias = nn.Parameter(torch.zeros(6))
            self.spatial_scale_logit = nn.Parameter(torch.tensor(-1.3862944))
        self._collect_window_diagnostics = False
        self._window_diagnostic_batches: list[dict[str, torch.Tensor]] = []

    @staticmethod
    def _match_spatial_residual(
        residual: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        residual_rms = residual.square().mean(dim=-1, keepdim=True).sqrt()
        target_rms = target.square().mean(dim=-1, keepdim=True).sqrt().detach()
        return residual * target_rms / residual_rms.clamp_min(1e-05)

    def _spatial_residual(self, fused: torch.Tensor):
        pooled = F.adaptive_avg_pool2d(fused, (self.num_channels, self.time_bins))
        tokens = pooled.permute(0, 3, 2, 1)
        query = self.spatial_q(tokens)
        key = self.spatial_k(tokens)
        value = self.spatial_v(tokens)
        logits = torch.einsum("btid,btjd->btij", query, key)
        logits = logits / math.sqrt(self.spatial_maps)
        logits = logits + self.spatial_relation_bias[self.relation_types][None, None]
        attention = torch.softmax(logits, dim=-1)
        message = torch.einsum("btij,btjd->btid", attention, value)
        raw = self.spatial_out(torch.cat([tokens, message], dim=-1))
        raw = self._match_spatial_residual(raw, tokens)
        gate = torch.sigmoid(self.spatial_gate(tokens)).squeeze(-1)
        scale = 0.5 * torch.sigmoid(self.spatial_scale_logit)
        delta = scale * gate[..., None] * raw
        delta = delta.permute(0, 3, 2, 1).contiguous()
        delta = delta.reshape(
            fused.shape[0] * self.spatial_maps * self.num_channels, 1, self.time_bins
        )
        delta = F.interpolate(
            delta, size=fused.shape[-1], mode="linear", align_corners=False
        )
        delta = delta.reshape(
            fused.shape[0], self.spatial_maps, self.num_channels, fused.shape[-1]
        )
        residual_ratio = delta.square().mean(
            dim=(1, 2, 3)
        ).sqrt() / fused.square().mean(dim=(1, 2, 3)).sqrt().clamp_min(1e-06)
        relation_entropy = -(
            attention.clamp_min(1e-08) * attention.clamp_min(1e-08).log()
        ).sum(dim=-1)
        relation_entropy = relation_entropy.mean(dim=(1, 2)) / math.log(28.0)
        consistency = (
            F.cosine_similarity(tokens, tokens + raw, dim=-1).mean(dim=(1, 2)) + 1.0
        ) * 0.5
        info = {
            "spatial_residual_ratio": residual_ratio,
            "spatial_relation_entropy": relation_entropy,
            "spatial_gate": gate.mean(dim=(1, 2)),
            "spatial_consistency": consistency,
            "spatial_scale": scale.expand(fused.shape[0]),
        }
        return (fused + delta, info)

    def _e096_fused(self, eeg: torch.Tensor):
        spectrum = torch.fft.rfft(eeg, n=self.time_samples, dim=-1)
        bands = torch.fft.irfft(
            spectrum[:, None, :, :] * self.band_masks[None, :, None, :],
            n=self.time_samples,
            dim=-1,
        )
        weights = torch.softmax(self.band_logits, dim=0)
        band_maps = []
        diagnostic_batches = []
        for index in range(len(self.BAND_NAMES)):
            scan = self._scan_encode(bands[:, index, :8])
            ear = self._ear_encode(bands[:, index, 8:])
            scan_tokens = self._tokens(scan, self.scan_channels)
            ear_tokens = self._tokens(ear, self.ear_channels)
            s2e, s2e_gate, s2e_entropy, s2e_consistency = self._route(
                ear_tokens, scan_tokens, self.scan_to_ear, self.scan_to_ear_scale_logit
            )
            e2s, e2s_gate, e2s_entropy, e2s_consistency = self._route(
                scan_tokens, ear_tokens, self.ear_to_scan, self.ear_to_scan_scale_logit
            )
            scan_residual = self._scatter_tokens(
                e2s, self.scan_channels, scan.shape[-1], 0, self.scan_channels
            )
            ear_residual = self._scatter_tokens(
                s2e, self.ear_channels, ear.shape[-1], 0, self.ear_channels
            )
            branch_map = self.band_branch_fusion(
                torch.cat([scan + scan_residual, ear + ear_residual], dim=2)
            )
            base_rms = branch_map.square().mean(dim=(1, 2, 3)).sqrt().clamp_min(1e-06)
            residual = torch.cat([scan_residual, ear_residual], dim=2)
            diagnostic_batches.append(
                {
                    "scan_to_ear_gate": s2e_gate,
                    "ear_to_scan_gate": e2s_gate,
                    "scan_to_ear_residual_ratio": s2e.square().mean(dim=(1, 2)).sqrt()
                    / base_rms,
                    "ear_to_scan_residual_ratio": e2s.square().mean(dim=(1, 2)).sqrt()
                    / base_rms,
                    "bidirectional_residual_ratio": residual.square()
                    .mean(dim=(1, 2, 3))
                    .sqrt()
                    / base_rms,
                    "scan_to_ear_attention_entropy": s2e_entropy,
                    "ear_to_scan_attention_entropy": e2s_entropy,
                    "connection_consistency": (s2e_consistency + e2s_consistency) * 0.5,
                    "band_weight": weights[index].expand(eeg.shape[0]),
                }
            )
            band_maps.append(branch_map * weights[index])
        fused = self.band_fusion(torch.cat(band_maps, dim=1))
        diagnostics = {
            key: torch.stack([item[key] for item in diagnostic_batches], dim=1).mean(
                dim=1
            )
            for key in diagnostic_batches[0]
        }
        diagnostics.update(
            {
                f"{name}_band_weight": weights[index].expand(eeg.shape[0])
                for index, name in enumerate(self.BAND_NAMES)
            }
        )
        diagnostics["band_fusion_entropy"] = (
            -(weights * weights.clamp_min(1e-08).log()).sum() / math.log(4.0)
        ).expand(eeg.shape[0])
        return (fused, diagnostics)

    def diagnostics(self) -> dict:
        payload = {
            "model_family": "E096_backbone_plus_montage_spatial_residual",
            "variant": self.variant,
            "spatial_module": (
                "relation_aware_electrode_token_residual_before_E096_spatial_collapse"
                if self.variant.endswith("_full")
                else "deleted"
            ),
            "backbone": "E096_unchanged_four_band_dual_branch_bidirectional_token_exchange",
            "spatial_relation_types": [
                "self",
                "scan_same_side",
                "scan_opposite_side",
                "ear_same_side",
                "ear_opposite_side",
                "scan_ear_cross_group",
            ],
            "group_definition": self.group_definition,
            "bands": {
                name: list(limits)
                for name, limits in zip(self.BAND_NAMES, self.BAND_LIMITS)
            },
            "band_weights": dict(
                zip(
                    self.BAND_NAMES,
                    torch.softmax(self.band_logits, dim=0).detach().cpu().tolist(),
                )
            ),
            "frontal_channels": self.scan_channels,
            "ear_channels": self.ear_channels,
            "input_window_samples": self.time_samples,
            "uses_trial_history": False,
            "uses_attention_label_as_input": False,
            "uses_speech_or_envelopes": False,
            "uses_probability_fusion": False,
            "spatial_ablation": "delete_only_new_spatial_residual",
        }
        if self.variant.endswith("_full"):
            payload["spatial_relation_bias"] = (
                self.spatial_relation_bias.detach().cpu().tolist()
            )
        return payload

    def forward(
        self, eeg: torch.Tensor, envelopes: torch.Tensor | None = None
    ) -> torch.Tensor:
        del envelopes
        fused, diagnostics = self._e096_fused(eeg)
        if self.variant.endswith("_full"):
            fused, spatial_info = self._spatial_residual(fused)
        else:
            zero = eeg.new_zeros(eeg.shape[0])
            spatial_info = {
                "spatial_residual_ratio": zero,
                "spatial_relation_entropy": zero,
                "spatial_gate": zero,
                "spatial_consistency": zero,
                "spatial_scale": zero,
            }
        diagnostics.update(spatial_info)
        self._record_diagnostics(diagnostics)
        values = self.spatial_collapse(fused).squeeze(2)
        values = self.temporal_refine(values)
        pooled = torch.cat(
            [values.mean(dim=-1), values.std(dim=-1, unbiased=False)], dim=1
        )
        return self.classifier(pooled)


class FAIE112ParallelOcularEvidenceNet(FAIE102SpatialResidualNet):
    VALID_VARIANTS = {"fai_e112_ocular_full", "fai_e112_ocular_none"}
    BAND_NAMES = FAIE096BranchInteractionNet.BAND_NAMES
    BAND_LIMITS = FAIE096BranchInteractionNet.BAND_LIMITS

    def __init__(self, num_channels: int, time_samples: int, variant: str):
        if num_channels != 28:
            raise ValueError("E112 scan+ear screen supports exactly 28 channels")
        if variant not in FAIE112ParallelOcularEvidenceNet.VALID_VARIANTS:
            raise ValueError(variant)
        super().__init__(num_channels, time_samples, "fai_e102_spatial_residual_none")
        self.variant = variant
        self.ocular_maps = 32
        if variant.endswith("_full"):
            self.ocular_temporal_branches = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(
                            1,
                            8,
                            kernel_size=(1, kernel),
                            padding=(0, kernel // 2),
                            bias=False,
                        ),
                        nn.GroupNorm(4, 8),
                        nn.GELU(),
                    )
                    for kernel in (3, 7, 15)
                ]
            )
            self.ocular_projection = nn.Sequential(
                nn.Conv2d(24, self.ocular_maps, kernel_size=1, bias=False),
                nn.GroupNorm(8, self.ocular_maps),
                nn.GELU(),
                nn.Conv2d(
                    self.ocular_maps,
                    self.ocular_maps,
                    kernel_size=(3, 1),
                    padding=(1, 0),
                    bias=False,
                ),
                nn.GroupNorm(8, self.ocular_maps),
                nn.GELU(),
            )
            self.ocular_channel_score = nn.Conv2d(
                self.ocular_maps, 1, kernel_size=1, bias=True
            )
            self.ocular_temporal_refine = nn.Sequential(
                nn.Conv1d(
                    self.ocular_maps,
                    self.ocular_maps,
                    kernel_size=9,
                    padding=4,
                    bias=False,
                ),
                nn.GroupNorm(8, self.ocular_maps),
                nn.GELU(),
                nn.Conv1d(
                    self.ocular_maps,
                    self.ocular_maps,
                    kernel_size=5,
                    padding=2,
                    bias=False,
                ),
                nn.GroupNorm(8, self.ocular_maps),
                nn.GELU(),
            )
            self.classifier = nn.Sequential(
                nn.LayerNorm(128 + 2 * self.ocular_maps),
                nn.Linear(128 + 2 * self.ocular_maps, 64),
                nn.GELU(),
                nn.Dropout(0.25),
                nn.Linear(64, 2),
            )

    def _parallel_ocular_features(
        self, bands: torch.Tensor, weights: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, n_bands, _, time = bands.shape
        scan = bands[:, :, :8, :].reshape(batch * n_bands, 1, 8, time)
        temporal = torch.cat(
            [branch(scan) for branch in self.ocular_temporal_branches], dim=1
        )
        features = self.ocular_projection(temporal)
        logits = self.ocular_channel_score(features).squeeze(1)
        channel_weights = torch.softmax(logits, dim=1)
        summary = (features * channel_weights.unsqueeze(1)).sum(dim=2)
        summary = summary.reshape(batch, n_bands, self.ocular_maps, time)
        summary = (summary * weights[None, :, None, None]).sum(dim=1)
        summary = self.ocular_temporal_refine(summary)
        pooled = torch.cat(
            [summary.mean(dim=-1), summary.std(dim=-1, unbiased=False)], dim=1
        )
        channel_weights = channel_weights.reshape(batch, n_bands, 8, time)
        entropy = -(
            channel_weights.clamp_min(1e-08) * channel_weights.clamp_min(1e-08).log()
        ).sum(dim=2).mean(dim=(1, 2)) / math.log(8.0)
        max_weight = channel_weights.max(dim=2).values.mean(dim=(1, 2))
        return (
            pooled,
            {
                "ocular_channel_entropy": entropy,
                "ocular_channel_max_weight": max_weight,
                "ocular_feature_ratio": pooled.square().mean(dim=1).sqrt(),
            },
        )

    def diagnostics(self) -> dict:
        payload = {
            "model_family": "E096_backbone_plus_parallel_current_window_ocular_evidence",
            "variant": self.variant,
            "backbone": "E096_unchanged_four_band_dual_branch_bidirectional_token_exchange",
            "new_module": (
                "parallel_scan_four_band_temporal_spatial_evidence_branch"
                if self.variant.endswith("_full")
                else "deleted"
            ),
            "ocular_channel_weighting": "learned_softmax_over_eight_scan_channels",
            "bands": {
                name: list(limits)
                for name, limits in zip(self.BAND_NAMES, self.BAND_LIMITS)
            },
            "input_window_samples": self.time_samples,
            "uses_trial_history": False,
            "uses_attention_label_as_input": False,
            "uses_speech_or_envelopes": False,
            "uses_probability_fusion": False,
            "ablation": "delete_only_parallel_ocular_evidence_branch",
        }
        return payload

    def forward(
        self, eeg: torch.Tensor, envelopes: torch.Tensor | None = None
    ) -> torch.Tensor:
        del envelopes
        batch = eeg.shape[0]
        fused, diagnostics = self._e096_fused(eeg)
        values = self.spatial_collapse(fused).squeeze(2)
        values = self.temporal_refine(values)
        pooled_base = torch.cat(
            [values.mean(dim=-1), values.std(dim=-1, unbiased=False)], dim=1
        )
        if self.variant.endswith("_full"):
            spectrum = torch.fft.rfft(eeg, n=self.time_samples, dim=-1)
            bands = torch.fft.irfft(
                spectrum[:, None, :, :] * self.band_masks[None, :, None, :],
                n=self.time_samples,
                dim=-1,
            )
            weights = torch.softmax(self.band_logits, dim=0)
            ocular, ocular_info = self._parallel_ocular_features(bands, weights)
            pooled = torch.cat([pooled_base, ocular], dim=1)
            diagnostics.update(ocular_info)
            diagnostics["ocular_feature_ratio"] = ocular.square().mean(
                dim=1
            ).sqrt() / pooled_base.square().mean(dim=1).sqrt().clamp_min(1e-06)
        else:
            pooled = pooled_base
            zero = eeg.new_zeros(batch)
            diagnostics.update(
                {
                    "ocular_channel_entropy": zero,
                    "ocular_channel_max_weight": zero,
                    "ocular_feature_ratio": zero,
                }
            )
        self._record_diagnostics(diagnostics)
        return self.classifier(pooled)


class FAIE115BilateralEvidenceInteractionNet(FAIE112ParallelOcularEvidenceNet):
    VALID_VARIANTS = {
        "fai_e115_bilateral_evidence_full",
        "fai_e115_bilateral_evidence_none",
    }
    BAND_NAMES = FAIE096BranchInteractionNet.BAND_NAMES
    BAND_LIMITS = FAIE096BranchInteractionNet.BAND_LIMITS

    def __init__(self, num_channels: int, time_samples: int, variant: str):
        if num_channels != 28:
            raise ValueError("E115 scan+ear screen supports exactly 28 channels")
        if variant not in FAIE115BilateralEvidenceInteractionNet.VALID_VARIANTS:
            raise ValueError(variant)
        super().__init__(num_channels, time_samples, "fai_e112_ocular_full")
        self.variant = variant
        self.auditory_maps = self.ocular_maps
        self.auditory_temporal_branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        1,
                        8,
                        kernel_size=(1, kernel),
                        padding=(0, kernel // 2),
                        bias=False,
                    ),
                    nn.GroupNorm(4, 8),
                    nn.GELU(),
                )
                for kernel in (3, 7, 15)
            ]
        )
        self.auditory_projection = nn.Sequential(
            nn.Conv2d(24, self.auditory_maps, kernel_size=1, bias=False),
            nn.GroupNorm(8, self.auditory_maps),
            nn.GELU(),
            nn.Conv2d(
                self.auditory_maps,
                self.auditory_maps,
                kernel_size=(3, 1),
                padding=(1, 0),
                bias=False,
            ),
            nn.GroupNorm(8, self.auditory_maps),
            nn.GELU(),
        )
        self.auditory_channel_score = nn.Conv2d(
            self.auditory_maps, 1, kernel_size=1, bias=True
        )
        self.auditory_temporal_refine = nn.Sequential(
            nn.Conv1d(
                self.auditory_maps,
                self.auditory_maps,
                kernel_size=9,
                padding=4,
                bias=False,
            ),
            nn.GroupNorm(8, self.auditory_maps),
            nn.GELU(),
            nn.Conv1d(
                self.auditory_maps,
                self.auditory_maps,
                kernel_size=5,
                padding=2,
                bias=False,
            ),
            nn.GroupNorm(8, self.auditory_maps),
            nn.GELU(),
        )
        self.auditory_band_logits = nn.Parameter(torch.zeros(4))
        evidence_dim = 2 * self.auditory_maps
        if variant.endswith("_full"):
            self.auditory_to_ocular = nn.Sequential(
                nn.LayerNorm(evidence_dim),
                nn.Linear(evidence_dim, evidence_dim),
                nn.GELU(),
                nn.Linear(evidence_dim, evidence_dim),
            )
            self.ocular_to_auditory = nn.Sequential(
                nn.LayerNorm(evidence_dim),
                nn.Linear(evidence_dim, evidence_dim),
                nn.GELU(),
                nn.Linear(evidence_dim, evidence_dim),
            )
            self.evidence_gate = nn.Sequential(
                nn.LayerNorm(2 * evidence_dim),
                nn.Linear(2 * evidence_dim, 32),
                nn.GELU(),
                nn.Linear(32, 2),
            )
            nn.init.constant_(self.evidence_gate[-1].bias, -1.0)
        self.classifier = nn.Sequential(
            nn.LayerNorm(128 + 2 * evidence_dim),
            nn.Linear(128 + 2 * evidence_dim, 96),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(96, 2),
        )

    def _auditory_evidence_features(
        self, bands: torch.Tensor, weights: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, n_bands, _, time = bands.shape
        evidence_by_band = []
        weights_by_band = []
        for index in range(n_bands):
            signal = bands[:, index, 8:, :].unsqueeze(1)
            temporal = torch.cat(
                [branch(signal) for branch in self.auditory_temporal_branches], dim=1
            )
            features = self.auditory_projection(temporal)
            logits = self.auditory_channel_score(features).squeeze(1)
            channel_weights = torch.softmax(logits, dim=1)
            summary = (features * channel_weights.unsqueeze(1)).sum(dim=2)
            summary = self.auditory_temporal_refine(summary)
            evidence_by_band.append(summary)
            weights_by_band.append(channel_weights)
        evidence = sum(
            (value * weights[index] for index, value in enumerate(evidence_by_band))
        )
        pooled = torch.cat(
            [evidence.mean(dim=-1), evidence.std(dim=-1, unbiased=False)], dim=1
        )
        channel_weights = torch.stack(weights_by_band, dim=1)
        entropy = -(
            channel_weights.clamp_min(1e-08) * channel_weights.clamp_min(1e-08).log()
        ).sum(dim=2).mean(dim=(1, 2)) / math.log(20.0)
        max_weight = channel_weights.max(dim=2).values.mean(dim=(1, 2))
        return (
            pooled,
            {
                "auditory_channel_entropy": entropy,
                "auditory_channel_max_weight": max_weight,
                "auditory_feature_ratio": pooled.square().mean(dim=1).sqrt(),
            },
        )

    def diagnostics(self) -> dict:
        scan_weights = torch.softmax(self.band_logits, dim=0).detach().cpu().tolist()
        auditory_weights = (
            torch.softmax(self.auditory_band_logits, dim=0).detach().cpu().tolist()
        )
        return {
            "model_family": "E112_carrier_plus_bilateral_current_window_evidence_interaction",
            "variant": self.variant,
            "backbone": "E096_E102_carrier_preserved",
            "new_module": (
                "scan_ocular_evidence_plus_ear_auditory_evidence_bidirectional_exchange"
                if self.variant.endswith("_full")
                else "bilateral_evidence_exchange_deleted"
            ),
            "interaction": "ocular_to_auditory_and_auditory_to_ocular_bounded_feature_residuals",
            "scan_channel_weighting": "softmax_over_eight_scan_electrodes",
            "ear_channel_weighting": "softmax_over_twenty_cEEgrid_channels",
            "bands": {
                name: list(limits)
                for name, limits in zip(self.BAND_NAMES, self.BAND_LIMITS)
            },
            "scan_band_weights": dict(zip(self.BAND_NAMES, scan_weights)),
            "ear_band_weights": dict(zip(self.BAND_NAMES, auditory_weights)),
            "input_window_samples": self.time_samples,
            "uses_trial_history": False,
            "uses_attention_label_as_input": False,
            "uses_speech_or_envelopes": False,
            "uses_probability_fusion": False,
            "ablation": "delete_bilateral_evidence_exchange_only",
        }

    def forward(
        self, eeg: torch.Tensor, envelopes: torch.Tensor | None = None
    ) -> torch.Tensor:
        del envelopes
        batch = eeg.shape[0]
        fused, diagnostics = self._e096_fused(eeg)
        values = self.spatial_collapse(fused).squeeze(2)
        values = self.temporal_refine(values)
        pooled_base = torch.cat(
            [values.mean(dim=-1), values.std(dim=-1, unbiased=False)], dim=1
        )
        spectrum = torch.fft.rfft(eeg, n=self.time_samples, dim=-1)
        bands = torch.fft.irfft(
            spectrum[:, None, :, :] * self.band_masks[None, :, None, :],
            n=self.time_samples,
            dim=-1,
        )
        scan_weights = torch.softmax(self.band_logits, dim=0)
        auditory_weights = torch.softmax(self.auditory_band_logits, dim=0)
        ocular, ocular_info = self._parallel_ocular_features(bands, scan_weights)
        auditory, auditory_info = self._auditory_evidence_features(
            bands, auditory_weights
        )
        if self.variant.endswith("_full"):
            gates = 0.05 + 0.9 * torch.sigmoid(
                self.evidence_gate(torch.cat([ocular, auditory], dim=1))
            )
            auditory_delta = self.auditory_to_ocular(auditory)
            ocular_delta = self.ocular_to_auditory(ocular)
            ocular_out = ocular + gates[:, 0:1] * auditory_delta
            auditory_out = auditory + gates[:, 1:2] * ocular_delta
            diagnostics["auditory_to_ocular_gate"] = gates[:, 0]
            diagnostics["ocular_to_auditory_gate"] = gates[:, 1]
            diagnostics["auditory_to_ocular_residual_ratio"] = (
                gates[:, 0:1] * auditory_delta
            ).square().mean(dim=1).sqrt() / ocular.square().mean(
                dim=1
            ).sqrt().clamp_min(
                1e-06
            )
            diagnostics["ocular_to_auditory_residual_ratio"] = (
                gates[:, 1:2] * ocular_delta
            ).square().mean(dim=1).sqrt() / auditory.square().mean(
                dim=1
            ).sqrt().clamp_min(
                1e-06
            )
        else:
            ocular_out, auditory_out = (ocular, auditory)
            zero = eeg.new_zeros(batch)
            diagnostics.update(
                {
                    "auditory_to_ocular_gate": zero,
                    "ocular_to_auditory_gate": zero,
                    "auditory_to_ocular_residual_ratio": zero,
                    "ocular_to_auditory_residual_ratio": zero,
                }
            )
        diagnostics.update(ocular_info)
        diagnostics.update(auditory_info)
        diagnostics["bilateral_evidence_consistency"] = (
            1.0 - F.cosine_similarity(ocular_out, auditory_out, dim=1).abs()
        )
        diagnostics["scan_ear_evidence_ratio"] = ocular_out.square().mean(
            dim=1
        ).sqrt() / auditory_out.square().mean(dim=1).sqrt().clamp_min(1e-06)
        self._record_diagnostics(diagnostics)
        pooled = torch.cat([pooled_base, ocular_out, auditory_out], dim=1)
        return self.classifier(pooled)


class FAIE121SpatialInteractionNet(FAIE115BilateralEvidenceInteractionNet):
    VALID_VARIANTS = {
        "fai_e121_spatial_full",
        "fai_e121_spatial_none",
        "fai_e121_projection_only",
        "fai_e121_interaction_fixed",
        "fai_e121_no_scan",
        "fai_e121_no_ear",
    }

    def __init__(self, num_channels: int, time_samples: int, variant: str):
        if num_channels not in {20, 28}:
            raise ValueError(
                "E121 supports 20-channel ear-only or 28-channel scan-ear views"
            )
        if variant not in self.VALID_VARIANTS:
            raise ValueError(variant)
        self.input_channels = int(num_channels)
        super().__init__(28, time_samples, "fai_e115_bilateral_evidence_full")
        self.variant = variant
        del self.auditory_to_ocular
        del self.ocular_to_auditory
        del self.evidence_gate
        if variant.endswith("_none"):
            for module_name in (
                "ocular_temporal_branches",
                "ocular_projection",
                "ocular_channel_score",
                "ocular_temporal_refine",
                "auditory_temporal_branches",
                "auditory_projection",
                "auditory_channel_score",
                "auditory_temporal_refine",
            ):
                delattr(self, module_name)
            del self.band_logits
            self.register_buffer("band_logits", torch.zeros(4))
            del self.auditory_band_logits
            self.register_buffer("auditory_band_logits", torch.zeros(4))
        elif variant == "fai_e121_no_scan":
            for module_name in (
                "ocular_temporal_branches",
                "ocular_projection",
                "ocular_channel_score",
                "ocular_temporal_refine",
            ):
                delattr(self, module_name)
            del self.band_logits
            self.register_buffer("band_logits", torch.zeros(4))
        elif variant == "fai_e121_no_ear":
            for module_name in (
                "auditory_temporal_branches",
                "auditory_projection",
                "auditory_channel_score",
                "auditory_temporal_refine",
            ):
                delattr(self, module_name)
            del self.auditory_band_logits
            self.register_buffer("auditory_band_logits", torch.zeros(4))
        carrier_dim = 128
        evidence_dim = 2 * self.ocular_maps
        scan_enabled = variant not in {"fai_e121_spatial_none", "fai_e121_no_scan"}
        ear_enabled = variant not in {"fai_e121_spatial_none", "fai_e121_no_ear"}
        interaction_enabled = variant in {
            "fai_e121_spatial_full",
            "fai_e121_interaction_fixed",
            "fai_e121_no_scan",
            "fai_e121_no_ear",
        }
        adaptive_gate_enabled = variant in {
            "fai_e121_spatial_full",
            "fai_e121_no_scan",
            "fai_e121_no_ear",
        }
        if scan_enabled:
            self.scan_to_space = nn.Sequential(
                nn.LayerNorm(evidence_dim),
                nn.Linear(evidence_dim, 128),
                nn.GELU(),
                nn.Linear(128, carrier_dim),
            )
        if ear_enabled:
            self.ear_to_space = nn.Sequential(
                nn.LayerNorm(evidence_dim),
                nn.Linear(evidence_dim, 128),
                nn.GELU(),
                nn.Linear(128, carrier_dim),
            )
        if interaction_enabled:
            joint_dim = 4 * carrier_dim
            self.spatial_interaction = nn.Sequential(
                nn.LayerNorm(joint_dim),
                nn.Linear(joint_dim, 160),
                nn.GELU(),
                nn.Linear(160, carrier_dim),
            )
        if adaptive_gate_enabled:
            self.spatial_gate = nn.Sequential(
                nn.LayerNorm(joint_dim),
                nn.Linear(joint_dim, 48),
                nn.GELU(),
                nn.Linear(48, 1),
            )
            nn.init.constant_(self.spatial_gate[-1].bias, -2.5)
        self.classifier = nn.Sequential(
            nn.LayerNorm(carrier_dim),
            nn.Linear(carrier_dim, 64),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(64, 2),
        )

    def diagnostics(self) -> dict:
        payload = super().diagnostics()
        if self.variant.endswith("_none"):
            ablation = "carrier_only"
            interaction = "none"
        elif self.variant.endswith("projection_only"):
            ablation = "separate_evidence_and_projection_without_interaction"
            interaction = "separate_evidence_projected_without_cross_modal_interaction"
        elif self.variant.endswith("interaction_fixed"):
            ablation = "interaction_with_fixed_gate_0.25"
            interaction = (
                "scan_and_ear_evidence_projected_into_common_space_then_interacted"
            )
        elif self.variant.endswith("no_scan"):
            ablation = "scan_evidence_path_removed"
            interaction = "ear_evidence_projected_with_joint_interaction"
        elif self.variant.endswith("no_ear"):
            ablation = "ear_evidence_path_removed"
            interaction = "scan_evidence_projected_with_joint_interaction"
        else:
            ablation = "interaction_with_learned_bounded_gate"
            interaction = (
                "scan_and_ear_evidence_projected_into_common_space_then_interacted"
            )
        scan_enabled = self.variant not in {"fai_e121_spatial_none", "fai_e121_no_scan"}
        ear_enabled = self.variant not in {"fai_e121_spatial_none", "fai_e121_no_ear"}
        interaction_enabled = self.variant in {
            "fai_e121_spatial_full",
            "fai_e121_interaction_fixed",
            "fai_e121_no_scan",
            "fai_e121_no_ear",
        }
        adaptive_gate_enabled = self.variant in {
            "fai_e121_spatial_full",
            "fai_e121_no_scan",
            "fai_e121_no_ear",
        }
        payload.update(
            {
                "model_family": "shared_carrier_plus_fronto_auricular_spatial_interaction",
                "variant": self.variant,
                "interaction": interaction,
                "ablation": ablation,
                "none_is_shared_backbone_only": self.variant.endswith("_none"),
                "module_components": {
                    "shared_carrier": True,
                    "scan_evidence": scan_enabled,
                    "ear_evidence": ear_enabled,
                    "scan_projection": scan_enabled,
                    "ear_projection": ear_enabled,
                    "common_space_projection": scan_enabled or ear_enabled,
                    "cross_modal_interaction": interaction_enabled,
                    "adaptive_gate": adaptive_gate_enabled,
                    "fixed_gate": (
                        0.25 if self.variant.endswith("_interaction_fixed") else None
                    ),
                },
                "supported_channel_views": "scan_ear_full, ear_only_zero_padded, ear_padded",
                "input_channels": self.input_channels,
                "scan_channel_softmax": "electrode_dimension",
                "ear_channel_softmax": "electrode_dimension",
            }
        )
        return payload

    def _as_full_layout(self, eeg: torch.Tensor) -> torch.Tensor:
        if eeg.shape[1] == 28:
            return eeg
        if eeg.shape[1] == 20:
            zeros = eeg.new_zeros((eeg.shape[0], 8, eeg.shape[-1]))
            return torch.cat([zeros, eeg], dim=1)
        raise ValueError(f"E121 expected 20 or 28 channels, got {eeg.shape[1]}")

    def forward(
        self, eeg: torch.Tensor, envelopes: torch.Tensor | None = None
    ) -> torch.Tensor:
        del envelopes
        eeg = self._as_full_layout(eeg)
        batch = eeg.shape[0]
        fused, diagnostics = self._e096_fused(eeg)
        values = self.spatial_collapse(fused).squeeze(2)
        values = self.temporal_refine(values)
        carrier = torch.cat(
            [values.mean(dim=-1), values.std(dim=-1, unbiased=False)], dim=1
        )
        spatial_variants = {
            "fai_e121_spatial_full",
            "fai_e121_projection_only",
            "fai_e121_interaction_fixed",
            "fai_e121_no_scan",
            "fai_e121_no_ear",
        }
        if self.variant in spatial_variants:
            spectrum = torch.fft.rfft(eeg, n=self.time_samples, dim=-1)
            bands = torch.fft.irfft(
                spectrum[:, None, :, :] * self.band_masks[None, :, None, :],
                n=self.time_samples,
                dim=-1,
            )
            scan_enabled = self.variant != "fai_e121_no_scan"
            ear_enabled = self.variant != "fai_e121_no_ear"
            evidence_dim = 2 * self.ocular_maps
            carrier_dim = carrier.shape[1]
            if scan_enabled:
                scan_weights = torch.softmax(self.band_logits, dim=0)
                ocular, ocular_info = self._parallel_ocular_features(
                    bands, scan_weights
                )
                scan_space = self.scan_to_space(ocular)
            else:
                ocular = eeg.new_zeros((batch, evidence_dim))
                scan_space = eeg.new_zeros((batch, carrier_dim))
                zero = eeg.new_zeros(batch)
                ocular_info = {
                    "ocular_channel_entropy": zero,
                    "ocular_channel_max_weight": zero,
                    "ocular_feature_ratio": zero,
                }
            if ear_enabled:
                ear_weights = torch.softmax(self.auditory_band_logits, dim=0)
                auditory, auditory_info = self._auditory_evidence_features(
                    bands, ear_weights
                )
                ear_space = self.ear_to_space(auditory)
            else:
                auditory = eeg.new_zeros((batch, evidence_dim))
                ear_space = eeg.new_zeros((batch, carrier_dim))
                zero = eeg.new_zeros(batch)
                auditory_info = {
                    "auditory_channel_entropy": zero,
                    "auditory_channel_max_weight": zero,
                    "auditory_feature_ratio": zero,
                }
            if self.variant.endswith("projection_only"):
                residual = scan_space + ear_space
                pooled = carrier + residual
                gate = eeg.new_zeros(batch)
            else:
                joint = torch.cat(
                    [
                        scan_space,
                        ear_space,
                        scan_space * ear_space,
                        (scan_space - ear_space).abs(),
                    ],
                    dim=1,
                )
                delta = self.spatial_interaction(joint)
                if self.variant in {
                    "fai_e121_spatial_full",
                    "fai_e121_no_scan",
                    "fai_e121_no_ear",
                }:
                    gate = 0.02 + 0.48 * torch.sigmoid(
                        self.spatial_gate(joint)
                    ).squeeze(1)
                else:
                    gate = eeg.new_full((batch,), 0.25)
                residual = gate.unsqueeze(1) * delta
                pooled = carrier + residual
            diagnostics.update(
                {
                    "spatial_interaction_gate": gate,
                    "spatial_interaction_residual_ratio": residual.square()
                    .mean(dim=1)
                    .sqrt()
                    / carrier.square().mean(dim=1).sqrt().clamp_min(1e-06),
                    "scan_space_contribution_ratio": scan_space.square()
                    .mean(dim=1)
                    .sqrt()
                    / carrier.square().mean(dim=1).sqrt().clamp_min(1e-06),
                    "ear_space_contribution_ratio": ear_space.square()
                    .mean(dim=1)
                    .sqrt()
                    / carrier.square().mean(dim=1).sqrt().clamp_min(1e-06),
                }
            )
        else:
            pooled = carrier
            zero = eeg.new_zeros(batch)
            ocular_info = {
                "ocular_channel_entropy": zero,
                "ocular_channel_max_weight": zero,
                "ocular_feature_ratio": zero,
            }
            auditory_info = {
                "auditory_channel_entropy": zero,
                "auditory_channel_max_weight": zero,
                "auditory_feature_ratio": zero,
            }
            diagnostics.update(
                {
                    "spatial_interaction_gate": zero,
                    "spatial_interaction_residual_ratio": zero,
                    "scan_space_contribution_ratio": zero,
                    "ear_space_contribution_ratio": zero,
                }
            )
        diagnostics.update(ocular_info)
        diagnostics.update(auditory_info)
        if self.variant in spatial_variants and self.variant not in {
            "fai_e121_no_scan",
            "fai_e121_no_ear",
        }:
            diagnostics["bilateral_evidence_consistency"] = (
                1.0 - F.cosine_similarity(ocular, auditory, dim=1).abs()
            )
        else:
            diagnostics["bilateral_evidence_consistency"] = eeg.new_zeros(batch)
        self._record_diagnostics(diagnostics)
        return self.classifier(pooled)


class FAIE126UnifiedCarrierNet(nn.Module):
    VALID_VARIANTS = {"fai_e126_unified_28ch_carrier"}
    BAND_NAMES = FAIE096BranchInteractionNet.BAND_NAMES
    BAND_LIMITS = FAIE096BranchInteractionNet.BAND_LIMITS

    def __init__(self, num_channels: int, time_samples: int, variant: str):
        super().__init__()
        if num_channels != 28:
            raise ValueError("E126 unified carrier supports exactly 28 channels")
        if variant not in self.VALID_VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        self.num_channels = int(num_channels)
        self.time_samples = int(time_samples)
        self.branch_maps = 8
        self.feature_maps = 32
        frequencies = torch.fft.rfftfreq(self.time_samples, d=1.0 / 128.0)
        self.register_buffer(
            "band_masks",
            torch.stack(
                [
                    ((frequencies >= low) & (frequencies < high)).float()
                    for low, high in self.BAND_LIMITS
                ]
            ),
        )
        self.band_logits = nn.Parameter(torch.zeros(len(self.BAND_NAMES)))
        self.shared_temporal_branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        1,
                        self.branch_maps,
                        kernel_size=(1, kernel),
                        padding=(0, kernel // 2),
                        bias=False,
                    ),
                    nn.GroupNorm(4, self.branch_maps),
                    nn.GELU(),
                )
                for kernel in (3, 5, 7, 11, 15, 23)
            ]
        )
        self.shared_projection = nn.Sequential(
            nn.Conv2d(
                6 * self.branch_maps, self.feature_maps, kernel_size=1, bias=False
            ),
            nn.GroupNorm(8, self.feature_maps),
            nn.GELU(),
            nn.Conv2d(
                self.feature_maps,
                self.feature_maps,
                kernel_size=(3, 15),
                padding=(1, 7),
                bias=False,
            ),
            nn.GroupNorm(8, self.feature_maps),
            nn.GELU(),
        )
        self.band_map_fusion = nn.Sequential(
            nn.Conv2d(self.feature_maps, 48, kernel_size=1, bias=False),
            nn.GroupNorm(8, 48),
            nn.GELU(),
        )
        self.band_fusion = nn.Sequential(
            nn.Conv2d(4 * 48, 64, kernel_size=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        self.spatial_collapse = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(28, 1), bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        self.temporal_refine = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=9, padding=4, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(64, 2),
        )
        self._collect_window_diagnostics = False
        self._window_diagnostic_batches: list[dict[str, torch.Tensor]] = []

    def set_diagnostic_collection(self, enabled: bool) -> None:
        self._collect_window_diagnostics = bool(enabled)
        if enabled:
            self._window_diagnostic_batches = []

    def _record_diagnostics(self, values: dict[str, torch.Tensor]) -> None:
        if self._collect_window_diagnostics:
            self._window_diagnostic_batches.append(
                {key: value.detach().cpu() for key, value in values.items()}
            )

    def consume_window_diagnostics(self) -> dict[str, list[float]]:
        if not self._window_diagnostic_batches:
            return {}
        keys = self._window_diagnostic_batches[0]
        payload = {
            key: torch.cat([batch[key] for batch in self._window_diagnostic_batches])
            .numpy()
            .astype(float)
            .tolist()
            for key in keys
        }
        self._window_diagnostic_batches = []
        return payload

    def diagnostics(self) -> dict:
        weights = torch.softmax(self.band_logits, dim=0).detach().cpu().tolist()
        return {
            "model_family": "unified_28_channel_multiband_carrier",
            "variant": self.variant,
            "architecture": "one_shared_encoder_for_all_28_channels",
            "temporal_kernels": [3, 5, 7, 11, 15, 23],
            "bands": {
                name: list(limits)
                for name, limits in zip(self.BAND_NAMES, self.BAND_LIMITS)
            },
            "band_weights": dict(zip(self.BAND_NAMES, weights)),
            "modality_specific_encoders": False,
            "bidirectional_token_interaction": False,
            "evidence_guided_refinement": False,
            "input_channels": self.num_channels,
            "input_window_samples": self.time_samples,
            "uses_trial_history": False,
            "uses_attention_label_as_input": False,
            "uses_speech_or_envelopes": False,
            "uses_probability_fusion": False,
        }

    def forward(
        self, eeg: torch.Tensor, envelopes: torch.Tensor | None = None
    ) -> torch.Tensor:
        del envelopes
        if eeg.shape[1] != self.num_channels:
            raise ValueError(
                f"E126 expected {self.num_channels} channels, got {eeg.shape[1]}"
            )
        spectrum = torch.fft.rfft(eeg, n=self.time_samples, dim=-1)
        bands = torch.fft.irfft(
            spectrum[:, None, :, :] * self.band_masks[None, :, None, :],
            n=self.time_samples,
            dim=-1,
        )
        weights = torch.softmax(self.band_logits, dim=0)
        band_maps = []
        for index in range(len(self.BAND_NAMES)):
            signal = bands[:, index].unsqueeze(1)
            temporal = torch.cat(
                [branch(signal) for branch in self.shared_temporal_branches], dim=1
            )
            encoded = self.shared_projection(temporal)
            band_maps.append(self.band_map_fusion(encoded) * weights[index])
        fused = self.band_fusion(torch.cat(band_maps, dim=1))
        values = self.spatial_collapse(fused).squeeze(2)
        values = self.temporal_refine(values)
        pooled = torch.cat(
            [values.mean(dim=-1), values.std(dim=-1, unbiased=False)], dim=1
        )
        self._record_diagnostics(
            {
                **{
                    f"{name}_band_weight": weights[index].expand(eeg.shape[0])
                    for index, name in enumerate(self.BAND_NAMES)
                },
                "band_fusion_entropy": (
                    -(weights * weights.clamp_min(1e-08).log()).sum()
                    / math.log(float(len(self.BAND_NAMES)))
                ).expand(eeg.shape[0]),
            }
        )
        return self.classifier(pooled)


class FAIE130DualBranchNoCDNet(FAIE096BranchInteractionNet):
    VALID_VARIANTS = {"fai_e130_dual_branch_no_cd_none"}

    def __init__(self, num_channels: int, time_samples: int, variant: str):
        if variant not in self.VALID_VARIANTS:
            raise ValueError(variant)
        super().__init__(num_channels, time_samples, "fai_e096_branch_interaction_none")
        self.variant = variant

    def _scan_encode(self, signal: torch.Tensor) -> torch.Tensor:
        features = torch.cat(
            [branch(signal.unsqueeze(1)) for branch in self.scan_temporal_branches],
            dim=1,
        )
        return self.scan_projection(features)

    def diagnostics(self) -> dict:
        payload = super().diagnostics()
        payload.update(
            {
                "model_family": "E096_dual_branch_no_frontal_common_difference",
                "variant": self.variant,
                "scan_feature": "multi_scale_frontal_features_without_common_difference",
                "frontal_common_difference_cue": "disabled_zero_contribution",
                "bidirectional_token_interaction": "disabled_zero_residual",
                "parameter_matched_to": "fai_e096_branch_interaction_none",
                "uses_left_right_partition": False,
            }
        )
        return payload
