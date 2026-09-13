from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalProjectionAttention(nn.Module):

    def __init__(self, feature_size: int, key_size: int = 8):
        super().__init__()
        self.query = nn.Linear(feature_size, key_size, bias=False)
        self.key = nn.Linear(feature_size, key_size, bias=False)
        self.value = nn.Linear(feature_size, feature_size, bias=False)
        self.scale = math.sqrt(key_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        query = torch.tanh(self.query(tokens))
        key = torch.tanh(self.key(tokens))
        value = torch.tanh(self.value(tokens))
        attention = torch.softmax(
            torch.matmul(query, key.transpose(-1, -2)) / self.scale, dim=-1
        )
        return torch.matmul(attention, value)


class STAnet(nn.Module):

    def __init__(self, num_channels: int):
        super().__init__()
        hidden = max(4, num_channels // 4)
        self.response = nn.Conv1d(1, 1, kernel_size=17, padding=8)
        self.spatial_gate = nn.Sequential(
            nn.Linear(num_channels, hidden),
            nn.ELU(),
            nn.Linear(hidden, num_channels),
            nn.Sigmoid(),
        )
        self.temporal_attention = TemporalProjectionAttention(num_channels, key_size=8)
        self.classifier = nn.Linear(num_channels, 2)

    def forward(
        self, eeg: torch.Tensor, envelopes: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch, channels, samples = eeg.shape
        response = F.elu(self.response(eeg.reshape(batch * channels, 1, samples)))
        response = response.amax(dim=-1).reshape(batch, channels)
        spatial_mask = self.spatial_gate(response)
        spatial = eeg * spatial_mask.unsqueeze(-1)
        attended = self.temporal_attention(spatial.transpose(1, 2))
        return self.classifier(attended.mean(dim=1))


class SinusoidalPositionEmbedding(nn.Module):

    def __init__(self, embedding_size: int, max_length: int = 512):
        super().__init__()
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, embedding_size, 2, dtype=torch.float32)
            * (-math.log(10000.0) / embedding_size)
        )
        values = torch.zeros(max_length, embedding_size)
        values[:, 0::2] = torch.sin(position * divisor)
        values[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("values", values.unsqueeze(0), persistent=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.values[:, : tokens.shape[1]]


class DARAttention(nn.Module):

    def __init__(self, embedding_size: int = 16, num_heads: int = 8):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.scale = embedding_size ** (-0.5)
        self.query = nn.Linear(embedding_size, embedding_size, bias=False)
        self.key = nn.Linear(embedding_size, embedding_size, bias=False)
        self.value = nn.Linear(embedding_size, embedding_size, bias=False)
        self.output_norm = nn.LayerNorm(embedding_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, length, _ = tokens.shape
        query = (
            self.query(tokens)
            .reshape(batch, length, self.num_heads, -1)
            .transpose(1, 2)
        )
        key = (
            self.key(tokens)
            .reshape(batch, length, self.num_heads, -1)
            .permute(0, 2, 3, 1)
        )
        value = (
            self.value(tokens)
            .reshape(batch, length, self.num_heads, -1)
            .transpose(1, 2)
        )
        weights = torch.softmax(torch.matmul(query, key) * self.scale, dim=-1)
        output = torch.matmul(weights, value).transpose(1, 2).reshape(batch, length, -1)
        return self.output_norm(output)


class DARRefinement(nn.Module):

    def __init__(self, embedding_size: int = 16):
        super().__init__()
        self.attention = DARAttention(embedding_size, num_heads=8)
        self.refine = nn.Sequential(
            nn.Conv1d(embedding_size, embedding_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(embedding_size),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.auxiliary = nn.Linear(embedding_size, 4)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.attention(tokens)
        tokens = self.refine(tokens.transpose(1, 2)).transpose(1, 2)
        auxiliary = self.auxiliary(tokens.mean(dim=1))
        return (tokens, auxiliary)


class DARNet(nn.Module):

    def __init__(self, num_channels: int, embedding_size: int = 16):
        super().__init__()
        self.temporal_embedding = nn.Sequential(
            nn.Conv2d(1, 4 * embedding_size, kernel_size=(1, 8), padding="same"),
            nn.BatchNorm2d(4 * embedding_size),
            nn.GELU(),
        )
        self.spatial_embedding = nn.Sequential(
            nn.Conv2d(
                4 * embedding_size, embedding_size, kernel_size=(num_channels, 1)
            ),
            nn.BatchNorm2d(embedding_size),
            nn.GELU(),
        )
        self.position = SinusoidalPositionEmbedding(embedding_size)
        self.refinement_1 = DARRefinement(embedding_size)
        self.refinement_2 = DARRefinement(embedding_size)
        self.classifier = nn.Linear(8, 2)

    def forward(
        self, eeg: torch.Tensor, envelopes: torch.Tensor | None = None
    ) -> torch.Tensor:
        del envelopes
        tokens = self.temporal_embedding(eeg.unsqueeze(1))
        tokens = self.spatial_embedding(tokens).squeeze(2).transpose(1, 2)
        tokens = tokens + self.position(tokens)
        tokens, auxiliary_1 = self.refinement_1(tokens)
        _, auxiliary_2 = self.refinement_2(tokens)
        return self.classifier(torch.cat([auxiliary_1, auxiliary_2], dim=-1))


class MultiScaleTemporalGate(nn.Module):

    def __init__(self, num_channels: int, time_samples: int):
        super().__init__()
        self.spatial = nn.Conv2d(1, 1, kernel_size=(num_channels, 1))
        self.branch_projection = nn.Conv1d(1, 3, kernel_size=1)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(1, 1, kernel_size=kernel, padding="same"),
                    nn.LayerNorm(time_samples),
                    nn.ELU(),
                    nn.AdaptiveAvgPool1d(1),
                )
                for kernel in (2, 4, 6)
            ]
        )
        self.output_projection = nn.Conv2d(1, num_channels, kernel_size=1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch = values.shape[0]
        reduced = self.spatial(values.permute(0, 2, 1, 3)).squeeze(2)
        branches = self.branch_projection(reduced).chunk(3, dim=1)
        gated = sum(
            (
                branch(branch_values) * branch_values
                for branch, branch_values in zip(self.branches, branches)
            )
        )
        return self.output_projection(gated.reshape(batch, 1, 1, -1))


class MHANetChannelAttention(nn.Module):

    def __init__(self, num_channels: int, time_samples: int):
        super().__init__()
        divisors = [value for value in range(1, 17) if num_channels % value == 0]
        self.num_heads = max(divisors)
        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))
        self.qkv = nn.Conv2d(num_channels, 3 * num_channels, kernel_size=1, bias=False)
        self.qkv_depthwise = nn.Conv2d(
            3 * num_channels,
            3 * num_channels,
            kernel_size=3,
            padding=1,
            groups=3 * num_channels,
            bias=False,
        )
        self.temporal_gate = MultiScaleTemporalGate(num_channels, time_samples)
        self.output_projection = nn.Conv2d(
            num_channels, num_channels, kernel_size=1, bias=False
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = values.shape
        query, key, value = self.qkv_depthwise(self.qkv(values)).chunk(3, dim=1)
        value = self.temporal_gate(value)
        channels_per_head = channels // self.num_heads
        query = query.reshape(batch, self.num_heads, channels_per_head, height * width)
        key = key.reshape(batch, self.num_heads, channels_per_head, height * width)
        value = value.reshape(batch, self.num_heads, channels_per_head, height * width)
        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)
        weights = torch.softmax(
            torch.matmul(query, key.transpose(-2, -1)) * self.temperature, dim=-1
        )
        output = torch.matmul(weights, value).reshape(batch, channels, height, width)
        return self.output_projection(output)


class MHANetGlobalAttention(nn.Module):

    def __init__(self):
        super().__init__()
        self.norm = nn.BatchNorm2d(1)
        self.expand = nn.Conv2d(1, 3, kernel_size=1)
        self.branches = nn.ModuleList(
            [
                nn.Conv2d(1, 1, kernel_size=kernel, dilation=dilation, padding="same")
                for kernel, dilation in [(3, 1), (5, 2), (7, 3)]
            ]
        )
        self.reduce = nn.Conv2d(3, 1, kernel_size=1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        shortcut = values
        expanded = self.expand(self.norm(values))
        attended = [
            branch(part) * part
            for branch, part in zip(self.branches, expanded.chunk(3, dim=1))
        ]
        return self.reduce(expanded * torch.cat(attended, dim=1)) + shortcut


class MHANet(nn.Module):

    def __init__(self, num_channels: int, time_samples: int):
        super().__init__()
        self.channel_attention = MHANetChannelAttention(num_channels, time_samples)
        self.global_attention = MHANetGlobalAttention()
        self.spatio_temporal = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=(1, 2)),
            nn.BatchNorm2d(5),
            nn.ELU(),
            nn.Conv2d(5, 5, kernel_size=(num_channels, 1)),
            nn.BatchNorm2d(5),
            nn.ELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(5, 2)

    def forward(
        self, eeg: torch.Tensor, envelopes: torch.Tensor | None = None
    ) -> torch.Tensor:
        del envelopes
        values = eeg.unsqueeze(1).permute(0, 2, 1, 3)
        values = self.channel_attention(values).permute(0, 2, 1, 3)
        values = self.global_attention(values)
        return self.classifier(self.spatio_temporal(values))


class SqueezeExcitation2d(nn.Module):

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.gate(values)


class HCAN(nn.Module):

    def __init__(self, num_channels: int, feature_size: int = 16):
        super().__init__()
        self.spatial_temporal = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(4, 64, kernel_size=(num_channels, 1)),
            nn.BatchNorm2d(64),
            nn.GELU(),
            SqueezeExcitation2d(64),
            nn.Conv2d(64, 64, kernel_size=(1, 7), padding=(0, 3), groups=64),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.feature_projection = nn.Conv1d(64, feature_size, kernel_size=1)
        self.scale_branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        feature_size,
                        feature_size,
                        kernel_size=kernel,
                        padding=kernel // 2,
                        groups=feature_size,
                    ),
                    nn.Conv1d(feature_size, feature_size, kernel_size=1),
                    nn.GELU(),
                )
                for kernel in (3, 5, 7)
            ]
        )
        self.branch_score = nn.Linear(feature_size, 1)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.local_attention = nn.Sequential(
            nn.Conv1d(
                feature_size,
                feature_size,
                kernel_size=3,
                padding=1,
                groups=feature_size,
            ),
            nn.Conv1d(feature_size, feature_size, kernel_size=1),
            nn.GELU(),
        )
        self.global_attention = nn.MultiheadAttention(
            feature_size, num_heads=4, dropout=0.1, batch_first=True
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(2 * feature_size, feature_size), nn.Sigmoid()
        )
        self.output_norm = nn.LayerNorm(feature_size)
        self.classifier = nn.Linear(feature_size, 2)

    def forward(
        self, eeg: torch.Tensor, envelopes: torch.Tensor | None = None
    ) -> torch.Tensor:
        del envelopes
        features = self.spatial_temporal(eeg.unsqueeze(1)).squeeze(2)
        features = self.feature_projection(features)
        branch_values = [branch(features) for branch in self.scale_branches]
        scores = torch.stack(
            [
                self.branch_score(branch.mean(dim=-1)).squeeze(-1)
                for branch in branch_values
            ],
            dim=1,
        )
        weights = torch.softmax(scores / self.temperature.clamp_min(0.05), dim=1)
        features = sum(
            (
                weights[:, index, None, None] * branch
                for index, branch in enumerate(branch_values)
            )
        )
        local = self.local_attention(features).transpose(1, 2)
        tokens = features.transpose(1, 2)
        global_values, _ = self.global_attention(
            tokens, tokens, tokens, need_weights=False
        )
        gate = self.fusion_gate(torch.cat([local, global_values], dim=-1))
        fused = self.output_norm(gate * local + (1.0 - gate) * global_values)
        return self.classifier(fused.mean(dim=1))

