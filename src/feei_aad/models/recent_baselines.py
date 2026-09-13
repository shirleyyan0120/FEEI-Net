"""Adapters for DBPNet, ListenNet, and XANet."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


RECENT_MODEL_PROVENANCE = {
    "dbpnet": {
        "implementation": "adapted_author_code",
        "repository": "https://github.com/fchest/DBPNet",
        "commit": "99a153fdbb8276f227ac6393529a24af9afc5dbc",
        "adaptation": (
            "Fold-local CSP; fixed custom-montage topographic projection; "
            "standard model input/output contract."
        ),
    },
    "listennet": {
        "implementation": "adapted_author_code",
        "repository": "https://github.com/fchest/ListenNet",
        "commit": "286b8ac4bddc7e6baee870940457809131249737",
        "adaptation": (
            "Fold-local Euclidean alignment; device-neutral initialization; "
            "standard model input/output contract."
        ),
    },
    "xanet": {
        "implementation": "paper_faithful_reproduction",
        "paper": "https://doi.org/10.1109/NER52421.2023.10123792",
        "adaptation": (
            "No public author implementation was located. The reproduction "
            "implements bilateral channel grouping and bidirectional "
            "cross-attention from the published method description."
        ),
    },
}


def bilateral_channel_groups(channel_view: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return participant-right and participant-left channel indices."""
    if channel_view == "frontal_only":
        return tuple(range(0, 4)), tuple(range(4, 8))
    if channel_view == "ear_only":
        return tuple(range(0, 10)), tuple(range(10, 20))
    if channel_view == "frontal_ear":
        return (
            tuple(range(0, 4)) + tuple(range(8, 18)),
            tuple(range(4, 8)) + tuple(range(18, 28)),
        )
    raise ValueError(f"Unsupported channel view: {channel_view}")


def _montage_coordinates(channel_view: str) -> np.ndarray:
    frontal = np.asarray(
        [
            (-0.16, 0.86),
            (-0.39, 0.83),
            (-0.62, 0.70),
            (-0.82, 0.43),
            (0.16, 0.86),
            (0.39, 0.83),
            (0.62, 0.70),
            (0.82, 0.43),
        ],
        dtype=np.float32,
    )
    angles = np.linspace(0.72 * np.pi, -0.72 * np.pi, 10, dtype=np.float32)
    right = np.column_stack(
        (-0.90 + 0.20 * np.cos(angles), 0.02 + 0.47 * np.sin(angles))
    )
    left = right[::-1].copy()
    left[:, 0] *= -1.0
    ear = np.vstack((right, left)).astype(np.float32)
    if channel_view == "frontal_only":
        return frontal
    if channel_view == "ear_only":
        return ear
    if channel_view == "frontal_ear":
        return np.vstack((frontal, ear))
    raise ValueError(f"Unsupported channel view: {channel_view}")


def _topographic_projection(channel_view: str, image_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    coordinates = _montage_coordinates(channel_view)
    axis = np.linspace(-1.18, 1.18, image_size, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(axis, axis, indexing="ij")
    grid = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    squared_distance = np.sum(
        (coordinates[:, None, :] - grid[None, :, :]) ** 2,
        axis=-1,
    )
    weights = np.exp(-squared_distance / (2.0 * 0.22**2))
    nearest = np.sqrt(squared_distance.min(axis=0))
    support = (nearest <= 0.42).astype(np.float32)
    weights /= np.maximum(weights.sum(axis=0, keepdims=True), 1e-8)
    weights *= support[None, :]
    return weights.astype(np.float32), support.reshape(image_size, image_size)


class TemporalAttentiveBranch(nn.Module):
    def __init__(self, channels: int, samples: int):
        super().__init__()
        self.scale = math.sqrt(channels)
        positions = torch.arange(samples, dtype=torch.float32)[:, None]
        frequencies = torch.exp(
            torch.arange(0, channels, 2, dtype=torch.float32)
            * (-math.log(10000.0) / max(channels, 1))
        )
        encoding = torch.zeros(samples, channels)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        if channels > 1:
            encoding[:, 1::2] = torch.cos(
                positions * frequencies[: encoding[:, 1::2].shape[1]]
            )
        self.register_buffer("position_encoding", encoding[None, :, :])
        self.pre_attention_norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            channels, num_heads=1, dropout=0.0, batch_first=True
        )
        self.pre_feedforward_norm = nn.LayerNorm(channels)
        self.feedforward = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
        )
        self.output_norm = nn.LayerNorm(channels)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(channels, 64, kernel_size=7),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc1 = nn.Linear(64, 16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        tokens = eeg.transpose(1, 2)
        tokens = self.scale * tokens + self.position_encoding[:, : tokens.shape[1]]
        normalized = self.pre_attention_norm(tokens)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        tokens = tokens + attended
        tokens = tokens + self.feedforward(self.pre_feedforward_norm(tokens))
        tokens = self.output_norm(tokens)
        features = self.temporal_conv(tokens.transpose(1, 2)).flatten(1)
        features = torch.sigmoid(self.fc1(features))
        features = F.dropout(features, p=0.8, training=self.training)
        return F.relu(self.fc2(features))


class FrequencyConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            (1, kernel_size, kernel_size),
            (1, stride, stride),
            (0, padding, padding),
            bias=False,
        )
        self.norm = nn.BatchNorm3d(out_channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.norm(self.conv(values))


class FrequencyResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        stride: int = 1,
    ):
        super().__init__()
        self.conv1 = FrequencyConvLayer(in_channels, hidden_channels, 1)
        self.conv2 = FrequencyConvLayer(
            hidden_channels, hidden_channels, 3, stride=stride, padding=1
        )
        self.conv3 = FrequencyConvLayer(hidden_channels, out_channels, 1)
        self.downsample = (
            FrequencyConvLayer(in_channels, out_channels, 1, stride=stride)
            if in_channels != out_channels
            else None
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        values = F.relu(self.conv1(values))
        values = F.relu(self.conv2(values))
        values = self.conv3(values)
        if self.downsample is not None:
            residual = self.downsample(residual)
        return F.dropout(F.relu(values + residual), p=0.3, training=self.training)


class FrequencyResidualBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = FrequencyConvLayer(1, 32, 7, stride=2, padding=3)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = FrequencyResidualBlock(32, 32, 64)
        self.conv2 = nn.Conv3d(64, 64, 3, padding=1)
        self.norm2 = nn.BatchNorm3d(64)
        self.layer2 = FrequencyResidualBlock(64, 64, 128, stride=2)
        self.conv3 = nn.Conv3d(128, 128, 3, padding=1)
        self.norm3 = nn.BatchNorm3d(128)
        self.layer3 = FrequencyResidualBlock(128, 128, 256, stride=2)
        self.conv5 = nn.Conv3d(256, 4, 1)
        self.norm5 = nn.BatchNorm3d(4)
        self.pool5 = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = torch.tanh(self.conv1(values))
        batch, channels, bands, height, width = values.shape
        values = self.pool1(values.reshape(batch, channels * bands, height, width))
        values = values.reshape(batch, channels, bands, values.shape[-2], values.shape[-1])
        values = self.layer1(values)
        values = F.relu(self.norm2(self.conv2(values)))
        values = self.layer2(values)
        values = F.relu(self.norm3(self.conv3(values)))
        values = self.layer3(values)
        values = F.relu(self.norm5(self.conv5(values)))
        return F.relu(self.pool5(values)).flatten(1)


class DBPNet(nn.Module):
    def __init__(
        self,
        sensor_channels: int,
        samples: int,
        channel_view: str,
        sampling_rate: int = 128,
    ):
        super().__init__()
        expected_channels = {
            "frontal_only": 8,
            "ear_only": 20,
            "frontal_ear": 28,
        }
        if expected_channels.get(channel_view) != sensor_channels:
            raise ValueError(
                f"DBPNet expected {expected_channels.get(channel_view)} channels "
                f"for {channel_view}, got {sensor_channels}"
            )
        self.sensor_channels = sensor_channels
        self.samples = samples
        self.temporal_branch = TemporalAttentiveBranch(sensor_channels, samples)
        self.frequency_branch = FrequencyResidualBranch()
        self.classifier = nn.Linear(8, 2)
        projection, support = _topographic_projection(channel_view)
        self.register_buffer("topographic_projection", torch.from_numpy(projection))
        self.register_buffer("topographic_support", torch.from_numpy(support))
        frequencies = torch.fft.rfftfreq(samples, d=1.0 / sampling_rate)
        masks = [
            (frequencies >= low) & (frequencies <= high)
            for low, high in ((1, 3), (4, 7), (8, 13), (14, 30), (31, 50))
        ]
        self.register_buffer("frequency_masks", torch.stack(masks))

    def _frequency_maps(self, raw: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(raw, n=self.samples, dim=-1)
        power = spectrum.real.square() + spectrum.imag.square()
        band_power = torch.stack(
            [
                power[..., mask].sum(dim=-1)
                for mask in self.frequency_masks
            ],
            dim=1,
        )
        band_features = torch.log2(band_power / self.samples + 1e-8)
        maps = torch.einsum(
            "bkc,cp->bkp", band_features, self.topographic_projection
        ).reshape(raw.shape[0], 5, 32, 32)
        support = self.topographic_support[None, None, :, :]
        supported_count = support.sum().clamp_min(1.0)
        mean = (maps * support).sum(dim=(1, 2, 3), keepdim=True) / (
            supported_count * maps.shape[1]
        )
        variance = (
            (maps - mean).square() * support
        ).sum(dim=(1, 2, 3), keepdim=True) / (
            supported_count * maps.shape[1]
        )
        maps = (maps - mean) / torch.sqrt(variance + 1e-6)
        return (maps * support).unsqueeze(1)

    def forward(
        self, eeg: torch.Tensor, envelopes: torch.Tensor | None = None
    ) -> torch.Tensor:
        del envelopes
        if eeg.shape[1] != 2 * self.sensor_channels:
            raise ValueError(
                "DBPNet expects [anatomical_raw, fold-local_CSP] input blocks"
            )
        raw = eeg[:, : self.sensor_channels]
        csp = eeg[:, self.sensor_channels :]
        temporal = self.temporal_branch(csp)
        frequency = self.frequency_branch(self._frequency_maps(raw))
        return self.classifier(torch.cat((temporal, frequency), dim=1))


class AlignChannels(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.projection = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels > out_channels
            else None
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if self.projection is not None:
            return self.projection(values)
        if self.in_channels < self.out_channels:
            return F.pad(
                values, (0, 0, 0, 0, 0, self.out_channels - self.in_channels)
            )
        return values


class DilatedInception(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        kernels = (1, 2, 3, 5)
        branch_channels = channels // len(kernels)
        self.branches = nn.ModuleList(
            nn.Conv2d(channels, branch_channels, (1, kernel), dilation=(1, 1))
            for kernel in kernels
        )
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        outputs = [branch(values) for branch in self.branches]
        minimum_time = min(output.shape[-1] for output in outputs)
        return self.norm(
            torch.cat([output[..., -minimum_time:] for output in outputs], dim=1)
        )


class CrossNestedAttention(nn.Module):
    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        if channels % groups:
            raise ValueError("CNA channels must be divisible by groups")
        self.groups = groups
        group_channels = channels // groups
        self.group_norm = nn.GroupNorm(group_channels, group_channels)
        self.fusion = nn.Conv2d(group_channels, group_channels, 1)
        self.sequence_reduce = nn.Conv1d(1, 1, kernel_size=1, stride=17)

    def _recalibrate(self, values: torch.Tensor) -> torch.Tensor:
        height_context = values.mean(dim=3, keepdim=True)
        time_context = values.mean(dim=2, keepdim=True).transpose(2, 3)
        fused = self.fusion(torch.cat((height_context, time_context), dim=2))
        height, time = values.shape[-2:]
        height_weight, time_weight = torch.split(fused, (height, time), dim=2)
        return self.group_norm(
            values
            * torch.sigmoid(height_weight)
            * torch.sigmoid(time_weight.transpose(2, 3))
        )

    def forward(self, temporal: torch.Tensor, spatial: torch.Tensor) -> torch.Tensor:
        batch, channels, temporal_height, time = temporal.shape
        spatial_height = spatial.shape[2]
        temporal_groups = temporal.reshape(
            batch * self.groups, channels // self.groups, temporal_height, time
        )
        spatial_groups = spatial.reshape(
            batch * self.groups, channels // self.groups, spatial_height, time
        )
        temporal_recalibrated = self._recalibrate(temporal_groups)
        spatial_recalibrated = self._recalibrate(spatial_groups)
        temporal_query = torch.softmax(
            temporal_recalibrated.mean(dim=(2, 3)).unsqueeze(1), dim=-1
        )
        spatial_query = torch.softmax(
            spatial_recalibrated.mean(dim=(2, 3)).unsqueeze(1), dim=-1
        )
        spatial_values = spatial_recalibrated.flatten(2)
        temporal_values = temporal_recalibrated.flatten(2)
        context = torch.cat(
            (
                torch.bmm(temporal_query, spatial_values),
                torch.bmm(spatial_query, temporal_values),
            ),
            dim=2,
        )
        weights = self.sequence_reduce(context)
        weights = weights.reshape(
            batch * self.groups, 1, spatial_height, time
        )
        output = spatial_groups * torch.sigmoid(weights)
        return output.reshape(batch, channels, spatial_height, time)


class ListenNet(nn.Module):
    def __init__(
        self,
        channels: int,
        samples: int,
        depth: int = 16,
        temporal_kernel: int = 8,
        average_pool: int = 8,
    ):
        super().__init__()
        self.channels = channels
        self.channel_weight = nn.Parameter(torch.empty(depth, depth))
        nn.init.xavier_uniform_(self.channel_weight)
        self.align = AlignChannels(channels, depth)
        self.temporal = nn.Sequential(
            nn.Conv2d(1, depth, (1, 1), bias=False),
            nn.BatchNorm2d(depth),
            nn.Conv2d(
                depth,
                depth,
                (1, temporal_kernel),
                groups=depth,
                bias=False,
            ),
            nn.BatchNorm2d(depth),
            nn.GELU(),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(depth, depth, 1, bias=False),
            nn.BatchNorm2d(depth),
            nn.Conv2d(depth, depth, (channels, 1), groups=depth, bias=False),
            nn.BatchNorm2d(depth),
            nn.GELU(),
        )
        self.multiscale = DilatedInception(depth)
        self.multiscale_skip = nn.Conv2d(
            depth, depth, (channels, 1), groups=depth, bias=True
        )
        self.merge_norm = nn.BatchNorm2d(depth)
        self.cna = CrossNestedAttention(depth, groups=8)
        feature_time = samples - temporal_kernel + 1
        pool = min(average_pool, feature_time)
        self.pool = nn.AvgPool2d((1, pool))
        pooled_time = (feature_time - pool) // pool + 1
        self.classifier = nn.Linear(depth * pooled_time, 2)
        self.dropout = 0.65
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def apply_constraints(self) -> None:
        last_weight = None
        for module in self.children():
            if hasattr(module, "weight") and not module.__class__.__name__.startswith(
                "BatchNorm"
            ):
                module.weight.data = torch.renorm(
                    module.weight.data, p=2, dim=0, maxnorm=2
                )
                last_weight = module.weight
        if last_weight is not None:
            last_weight.data = torch.renorm(
                last_weight.data, p=2, dim=0, maxnorm=0.5
            )

    def forward(
        self, eeg: torch.Tensor, envelopes: torch.Tensor | None = None
    ) -> torch.Tensor:
        del envelopes
        values = eeg.unsqueeze(1)
        temporal = self.temporal(values)
        spatial = self.spatial(temporal)
        multiscale = self.multiscale(temporal)
        skip = self.multiscale_skip(
            F.dropout(multiscale, self.dropout, training=self.training)
        )
        skip = F.interpolate(
            skip, size=(1, spatial.shape[-1]), mode="bilinear", align_corners=False
        )
        spatial = self.merge_norm(skip + spatial)
        aligned = self.align(temporal.permute(0, 2, 1, 3))
        aligned = torch.einsum("bdcw,sc->bdsw", aligned, self.channel_weight)
        features = self.cna(aligned, spatial)
        features = F.dropout(
            self.pool(features), p=self.dropout, training=self.training
        ).flatten(1)
        return self.classifier(features)


class BilateralTemporalEncoder(nn.Module):
    def __init__(self, token_dim: int = 64, temporal_bins: int = 8):
        super().__init__()
        self.temporal_bins = temporal_bins
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, (1, 9), padding=(0, 4), bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(
                16, 32, (1, 7), padding=(0, 3), groups=16, bias=False
            ),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.projection = nn.Linear(32 * temporal_bins, token_dim)
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        features = self.features(values.unsqueeze(1))
        features = F.adaptive_avg_pool2d(
            features, (values.shape[1], self.temporal_bins)
        )
        tokens = features.permute(0, 2, 1, 3).flatten(2)
        return self.norm(self.projection(tokens))


class BidirectionalCrossAttention(nn.Module):
    def __init__(self, embedding_dim: int = 64, heads: int = 4):
        super().__init__()
        self.right_to_left = nn.MultiheadAttention(
            embedding_dim, heads, dropout=0.1, batch_first=True
        )
        self.left_to_right = nn.MultiheadAttention(
            embedding_dim, heads, dropout=0.1, batch_first=True
        )
        self.left_norm1 = nn.LayerNorm(embedding_dim)
        self.right_norm1 = nn.LayerNorm(embedding_dim)
        self.left_norm2 = nn.LayerNorm(embedding_dim)
        self.right_norm2 = nn.LayerNorm(embedding_dim)
        self.left_feedforward = nn.Sequential(
            nn.Linear(embedding_dim, 2 * embedding_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2 * embedding_dim, embedding_dim),
        )
        self.right_feedforward = nn.Sequential(
            nn.Linear(embedding_dim, 2 * embedding_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2 * embedding_dim, embedding_dim),
        )

    def forward(
        self, left: torch.Tensor, right: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        left_query = self.left_norm1(left)
        right_query = self.right_norm1(right)
        left_context, _ = self.right_to_left(
            left_query, right_query, right_query, need_weights=False
        )
        right_context, _ = self.left_to_right(
            right_query, left_query, left_query, need_weights=False
        )
        left = left + left_context
        right = right + right_context
        left = left + self.left_feedforward(self.left_norm2(left))
        right = right + self.right_feedforward(self.right_norm2(right))
        return left, right


class XANet(nn.Module):
    def __init__(
        self,
        channels: int,
        channel_view: str,
        embedding_dim: int = 64,
    ):
        super().__init__()
        right, left = bilateral_channel_groups(channel_view)
        if len(right) + len(left) != channels:
            raise ValueError(
                f"XANet bilateral groups cover {len(right) + len(left)} of "
                f"{channels} channels"
            )
        self.register_buffer("right_indices", torch.tensor(right, dtype=torch.long))
        self.register_buffer("left_indices", torch.tensor(left, dtype=torch.long))
        self.encoder = BilateralTemporalEncoder(embedding_dim)
        self.right_position = nn.Parameter(
            torch.zeros(1, len(right), embedding_dim)
        )
        self.left_position = nn.Parameter(
            torch.zeros(1, len(left), embedding_dim)
        )
        nn.init.normal_(self.right_position, std=0.02)
        nn.init.normal_(self.left_position, std=0.02)
        self.cross_attention = BidirectionalCrossAttention(embedding_dim, heads=4)
        self.output_norm = nn.LayerNorm(2 * embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(2 * embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(embedding_dim, 2),
        )

    def forward(
        self, eeg: torch.Tensor, envelopes: torch.Tensor | None = None
    ) -> torch.Tensor:
        del envelopes
        right = self.encoder(eeg.index_select(1, self.right_indices))
        left = self.encoder(eeg.index_select(1, self.left_indices))
        right = right + self.right_position
        left = left + self.left_position
        left, right = self.cross_attention(left, right)
        pooled = self.output_norm(
            torch.cat((left.mean(dim=1), right.mean(dim=1)), dim=1)
        )
        return self.classifier(pooled)


def build_recent_baseline(
    model_id: str,
    model_input_channels: int,
    samples: int,
    channel_view: str,
) -> nn.Module:
    if model_id == "dbpnet":
        if model_input_channels % 2:
            raise ValueError("DBPNet hybrid input must contain two equal blocks")
        return DBPNet(model_input_channels // 2, samples, channel_view)
    if model_id == "listennet":
        return ListenNet(model_input_channels, samples)
    if model_id == "xanet":
        return XANet(model_input_channels, channel_view)
    raise ValueError(f"Unknown recent baseline: {model_id}")


def validate_bilateral_groups(
    channel_view: str, channels: int
) -> dict[str, Sequence[int]]:
    right, left = bilateral_channel_groups(channel_view)
    if set(right) & set(left):
        raise RuntimeError("Bilateral groups overlap")
    if set(right) | set(left) != set(range(channels)):
        raise RuntimeError("Bilateral groups do not cover the input")
    return {"participant_right": right, "participant_left": left}
