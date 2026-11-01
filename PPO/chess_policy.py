import math

import torch
import torch.nn as nn
from torch.distributions import Categorical

from PPO.chess_env import NUM_ACTIONS, NUM_PLANES

MASKED_LOGIT = -1e9


def _conv(c_in, c_out, k):
    return nn.Conv2d(c_in, c_out, k, padding=k // 2, bias=False)


class SEBlock(nn.Module):
    """Leela-style Squeeze-and-Excitation: gives 3x3 conv blocks global board context."""
    def __init__(self, channels, se_ratio=4):
        super().__init__()
        mid = max(channels // se_ratio, 16)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels * 2)

    def forward(self, x):
        # x: (B, C, 8, 8)
        w = x.mean(dim=(-2, -1))                      #Global Average Pool -> (B, C)
        w = torch.relu(self.fc1(w))
        w = self.fc2(w).unsqueeze(-1).unsqueeze(-1)   # (B, 2*C, 1, 1)
        scale, shift = w.chunk(2, dim=1)
        return torch.sigmoid(scale) * x + shift


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = _conv(channels, channels, 3)
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = _conv(channels, channels, 3)
        self.norm2 = nn.GroupNorm(8, channels)
        self.se = SEBlock(channels)

    def forward(self, x):
        out = torch.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.se(out)
        return torch.relu(x + out)


class ChessNet(nn.Module):
    discrete = True

    def __init__(self, channels=128, blocks=8, amp=False):
        super().__init__()
        self.config = {'channels': channels, 'blocks': blocks}
        self.amp = amp

        # Register 2 static coordinate planes (rank and file in [-1, 1]) so convs know board geometry
        coords = torch.linspace(-1.0, 1.0, 8)
        rank_plane = coords.view(1, 1, 8, 1).expand(1, 1, 8, 8)
        file_plane = coords.view(1, 1, 1, 8).expand(1, 1, 8, 8)
        self.register_buffer('coord_planes', torch.cat([rank_plane, file_plane], dim=1), persistent=False)

        self.stem = nn.Sequential(
            _conv(NUM_PLANES + 2, channels, 3),
            nn.GroupNorm(8, channels),
            nn.ReLU(),
        )
        self.tower = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])

        self.policy_head = nn.Sequential(
            _conv(channels, channels, 3),
            nn.GroupNorm(8, channels),
            nn.ReLU(),
            nn.Conv2d(channels, NUM_ACTIONS // 64, 1),
        )
        self.value_head = nn.Sequential(
            _conv(channels, 32, 1),
            nn.GroupNorm(4, 32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 64, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, math.sqrt(2.))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Zero-init the last conv of each residual block so blocks start as identity
        for block in self.tower:
            nn.init.zeros_(block.fc2.weight if hasattr(block, 'fc2') else block.se.fc2.weight)
            nn.init.zeros_(block.se.fc2.bias)

        nn.init.orthogonal_(self.policy_head[-1].weight, 0.01)  # near-uniform policy at start
        nn.init.orthogonal_(self.value_head[-1].weight, 0.01)   # near-zero value predictions at start

    def _encode(self, x):
        if x.dtype == torch.uint8:
            # Safe scaling whether uint8 is in {0, 1} or {0, 255}
            x = x.float()
            if x.max() > 1.0:
                x = x / 255.0
        coords = self.coord_planes.expand(x.shape[0], -1, -1, -1)
        x = torch.cat([x, coords], dim=1)
        return self.tower(self.stem(x))

    def _features(self, x):
        use_amp = self.amp and x.device.type == 'cuda'
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=use_amp):
            h = self._encode(x)
            logits = self.policy_head(h)
            value = self.value_head(h)
        logits = logits.float().permute(0, 2, 3, 1).reshape(x.shape[0], NUM_ACTIONS)
        return logits, value.float().squeeze(-1)

    def raw_outputs(self, x):
        """(unmasked action logits, value) as plain tensors."""
        return self._features(x)

    def forward(self, x, mask=None):
        logits, value = self._features(x)
        if mask is not None:
            logits = logits.masked_fill(~mask, MASKED_LOGIT)
        return Categorical(logits=logits, validate_args=False), value

    def get_value(self, x):
        use_amp = self.amp and x.device.type == 'cuda'
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=use_amp):
            h = self._encode(x)
            value = self.value_head(h)
        return value.float().squeeze(-1)

    def act(self, x, mask=None, deterministic=False):
        dist, value = self(x, mask)
        action = dist.mode if deterministic else dist.sample()
        return action, dist.log_prob(action), value

    def evaluate(self, x, action, mask=None):
        dist, value = self(x, mask)
        return dist.log_prob(action), dist.entropy(), value