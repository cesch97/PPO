import math

import torch
import torch.nn as nn
from torch.distributions import Categorical

from PPO.chess_env import NUM_ACTIONS, NUM_PLANES

MASKED_LOGIT = -1e9


def _conv(c_in, c_out, k):
    return nn.Conv2d(c_in, c_out, k, padding=k // 2, bias=False)


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = _conv(channels, channels, 3)
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = _conv(channels, channels, 3)
        self.norm2 = nn.GroupNorm(8, channels)

    def forward(self, x):
        out = torch.relu(self.norm1(self.conv1(x)))
        return torch.relu(x + self.norm2(self.conv2(out)))


class ChessNet(nn.Module):
    """
    Residual convolutional actor-critic for chess, with the same interface as `PPO.policy.ActorCritic` plus the mask
    of the legal actions.

    Input: uint8 planes (B, NUM_PLANES, 8, 8) seen from the player to move. Output: logits over the 4672 moves
    (policy head = 73 move types for each of the 64 from-squares, so the output keeps the board geometry) and the value
    of the position for the player to move (about the expected result in [-1, 1]).
    GroupNorm is used instead of BatchNorm: its output does not depend on the batch, which keeps the log-probabilities
    of the rollout and of the update consistent.
    """
    discrete = True

    def __init__(self, channels=64, blocks=6, amp=False):
        super().__init__()
        self.config = {'channels': channels, 'blocks': blocks}
        self.amp = amp
        self.stem = nn.Sequential(_conv(NUM_PLANES, channels, 3), nn.GroupNorm(8, channels), nn.ReLU())
        self.tower = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.policy_head = nn.Sequential(_conv(channels, channels, 3), nn.GroupNorm(8, channels), nn.ReLU(),
                                         nn.Conv2d(channels, NUM_ACTIONS // 64, 1))
        self.value_head = nn.Sequential(_conv(channels, 32, 1), nn.ReLU(), nn.Flatten(), nn.Linear(32 * 64, 256),
                                        nn.ReLU(), nn.Linear(256, 1))
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, math.sqrt(2.))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.policy_head[-1].weight, 0.01)  # near uniform policy at the start
        nn.init.orthogonal_(self.value_head[-1].weight, 1.)

    def _features(self, x):
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        use_amp = self.amp and x.device.type == 'cuda'
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=use_amp):
            h = self.tower(self.stem(x))
            logits = self.policy_head(h)
            value = self.value_head(h)
        # (B, 73, 8, 8) -> (B, 8, 8, 73) -> (B, 4672): index = from_square * 73 + move_type
        logits = logits.float().permute(0, 2, 3, 1).reshape(x.shape[0], NUM_ACTIONS)
        return logits, value.float().squeeze(-1)

    def forward(self, x, mask=None):
        """Returns (Categorical over the legal moves, value)."""
        logits, value = self._features(x)
        if mask is not None:
            logits = logits.masked_fill(~mask, MASKED_LOGIT)
        return Categorical(logits=logits, validate_args=False), value

    def get_value(self, x):
        return self._features(x)[1]

    def act(self, x, mask=None, deterministic=False):
        """Returns (action, log_prob, value) for a batch of observations."""
        dist, value = self(x, mask)
        action = dist.mode if deterministic else dist.sample()
        return action, dist.log_prob(action), value

    def evaluate(self, x, action, mask=None):
        """Returns (log_prob, entropy, value) of the given actions."""
        dist, value = self(x, mask)
        return dist.log_prob(action), dist.entropy(), value
