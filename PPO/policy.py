import math
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


def init_extractor_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight, math.sqrt(2.))
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, math.sqrt(2.))
        nn.init.zeros_(m.bias)

def init_crtic_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, 1)
        nn.init.zeros_(m.bias)

def init_actor_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, 0.01)
        nn.init.zeros_(m.bias)


class CnnExtractor(nn.Module):
    def __init__(self, input_shape):
        super(CnnExtractor, self).__init__()
        self.input_shape = input_shape

        self.cnn = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.cnn_flatten = self.feature_size()

        self.fc = nn.Sequential(nn.Linear(self.cnn_flatten, 512),
                                nn.ReLU())
        self.linear_feature = 512

    def forward(self, x):
        # uint8 frames are scaled here so they can be stored and transferred in their compact form
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        return self.fc(self.cnn(x).reshape(-1, self.cnn_flatten))

    def feature_size(self):
        with torch.no_grad():
            return self.cnn(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)


class MlpExtractor(nn.Module):
    def __init__(self, n_inputs, h_layer=64):
        super(MlpExtractor, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(n_inputs, h_layer),
            nn.ReLU(),
            nn.Linear(h_layer, h_layer),
            nn.ReLU()
        )
        self.linear_feature = h_layer

    def forward(self, x):
        return self.fc(x)


class ActorCritic(nn.Module):
    def __init__(self, input_shape, num_outputs, discrete, log_std=0):
        super(ActorCritic, self).__init__()

        if len(input_shape) == 3:
            self.feature_ext = CnnExtractor(input_shape)
        elif len(input_shape) == 1:
            self.feature_ext = MlpExtractor(input_shape[0])
        self.feature_ext.apply(init_extractor_weights)

        self.critic = nn.Linear(self.feature_ext.linear_feature, 1)
        self.actor = nn.Linear(self.feature_ext.linear_feature, num_outputs)
        self.critic.apply(init_crtic_weights)
        self.actor.apply(init_actor_weights)

        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * log_std)
        self.std_coef = nn.Parameter(torch.tensor(1).float(), requires_grad=False)
        self.discrete = discrete

    def forward(self, x):
        features = self.feature_ext(x)
        return self.distribution(self.actor(features)), self.critic(features).squeeze(-1)

    def distribution(self, out):
        if self.discrete:
            return Categorical(logits=out)
        std = self.log_std.exp().expand_as(out) * self.std_coef.abs()
        return Normal(out, std)

    def get_value(self, x):
        return self.critic(self.feature_ext(x)).squeeze(-1)

    def act(self, x, deterministic=False):
        """Return (action, log_prob, value) for a batch of observations."""
        dist, value = self(x)
        if deterministic:
            action = dist.mode
        else:
            action = dist.sample()
        return action, self.log_prob(dist, action), value

    def evaluate(self, x, action):
        """Return (log_prob, entropy, value) of the given actions."""
        dist, value = self(x)
        return self.log_prob(dist, action), self.entropy(dist), value

    def log_prob(self, dist, action):
        log_prob = dist.log_prob(action)
        return log_prob if self.discrete else log_prob.sum(-1)

    def entropy(self, dist):
        entropy = dist.entropy()
        return entropy if self.discrete else entropy.sum(-1)
