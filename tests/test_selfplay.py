import numpy as np
import torch

from PPO.chess_env import ChessVecEnv, NUM_ACTIONS, NUM_PLANES
from PPO.chess_policy import ChessNet
from PPO.selfplay import EnginePlayer, NetPlayer, SelfPlayPPO, negamax_gae, play_match


def test_negamax_gae_matches_the_recursive_definition():
    rng = np.random.default_rng(0)
    T, N, gamma = 12, 3, 0.9
    rewards = torch.tensor(rng.normal(size=(T, N)), dtype=torch.float32)
    values = torch.tensor(rng.normal(size=(T, N)), dtype=torch.float32)
    next_value = torch.tensor(rng.normal(size=N), dtype=torch.float32)
    masks = torch.ones(T, N)
    masks[5, 1] = 0  # game over after step 5 in env 1

    # lambda = 1: advantage = discounted negamax return - value
    adv = negamax_gae(rewards, values, masks, next_value, gamma, 1.0)
    for n in range(N):
        ret = next_value[n].item()
        for t in reversed(range(T)):
            ret = rewards[t, n].item() - gamma * ret * masks[t, n].item()
            assert abs(adv[t, n].item() - (ret - values[t, n].item())) < 1e-5

    # lambda = 0: one step TD error
    adv0 = negamax_gae(rewards, values, masks, next_value, gamma, 0.0)
    for t in range(T - 1):
        expected = rewards[t] - gamma * values[t + 1] * masks[t] - values[t]
        assert torch.allclose(adv0[t], expected, atol=1e-6)


def test_network_only_samples_legal_moves():
    net = ChessNet(channels=16, blocks=1)
    env = ChessVecEnv(32, seed=0)
    env.reset()
    obs, mask = torch.from_numpy(env.obs), torch.from_numpy(env.mask)
    action, log_prob, value = net.act(obs, mask)
    assert mask[torch.arange(32), action].all()
    log_prob2, entropy, value2 = net.evaluate(obs, action, mask)
    assert torch.allclose(log_prob, log_prob2) and torch.isfinite(entropy).all() and value.shape == (32,)
    # the policy is almost uniform over the legal moves at the start
    assert abs(entropy.mean().item() - torch.log(mask.sum(1).float()).mean().item()) < 0.1


def test_short_training_run_and_checkpoint(tmp_path):
    agent = SelfPlayPPO(device='cpu', channels=16, blocks=1)
    file = tmp_path / 'chess.pth'
    agent.learn(max_frames=4 * 8 * 8, num_envs=8, num_steps=8, batch_size=32, ppo_epochs=1, max_plies=30,
                snapshot_every=1, log_every=2, log_file=str(file), eval_games=4, eval_depths=(0, 1),
                p_self=0.4, p_pool=0.3, p_engine=0.3, plot_progress=False, seed=0, num_threads=2)
    assert agent.frame_idx == 256 and len(agent.history) == 2 and len(agent.pool) >= 1
    loaded = SelfPlayPPO.load(str(file), device='cpu')
    assert loaded.frame_idx == agent.frame_idx and len(loaded.pool) == len(agent.pool)
    for a, b in zip(agent.policy.parameters(), loaded.policy.parameters()):
        assert torch.equal(a, b)
    loaded.learn(max_frames=agent.frame_idx + 64, num_envs=8, num_steps=8, batch_size=32, ppo_epochs=1, max_plies=30,
                 eval_games=0, plot_progress=False, seed=1, num_threads=2)
    assert loaded.frame_idx == agent.frame_idx + 64


def test_play_match_scores():
    result = play_match(EnginePlayer(2), EnginePlayer(0), n_games=8, max_plies=120, seed=0)
    assert result['wins'] + result['draws'] + result['losses'] == 8
    assert result['wins'] >= 6  # a 2-ply search crushes a random mover
    agent = SelfPlayPPO(device='cpu', channels=16, blocks=1)
    result = play_match(agent.player(), EnginePlayer(0), n_games=4, max_plies=40, seed=0, record_game=True)
    assert len(result['moves']) <= 40 and all(len(m) == 3 for m in result['moves'])
