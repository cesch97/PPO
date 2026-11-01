import gymnasium as gym
import matplotlib
import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from PPO.PPO import PPO
from PPO.selfplay import SelfPlayPPO

matplotlib.use('Agg')


def read_events(log_dir):
    accumulator = EventAccumulator(str(log_dir), size_guidance={'histograms': 0, 'tensors': 0, 'images': 0})
    accumulator.Reload()
    return accumulator


def make_cartpole():
    return gym.make('CartPole-v1')


def test_ppo_logs_to_tensorboard(tmp_path):
    ppo = PPO(lambda: make_cartpole, device='cpu')
    ppo.learn(max_frames=4 * 16 * 8, num_envs=4, num_steps=16, batch_size=32, ppo_epochs=1, log_every=2,
              tensorboard=tmp_path)
    events = read_events(tmp_path)
    scalars, tensors = events.Tags()['scalars'], events.Tags()['tensors']
    for tag in ('train/loss', 'train/grad_norm', 'train/approx_kl', 'rollout/explained_variance',
                'rollout/ep_return_mean', 'eval/reward', 'norms/global_weights', 'norms/global_grads',
                'model/num_parameters'):
        assert tag in scalars, tag
    assert len(events.Scalars('train/loss')) == 8 and len(events.Scalars('eval/reward')) == 4
    assert events.Graph()
    assert 'weights/feature_ext/fc/0/weight' in events.Tags()['histograms']
    assert 'grads/actor/weight' in events.Tags()['histograms']
    for tag in ('setup/hyperparameters', 'model/architecture', 'model/summary', 'model/parameters'):
        assert tag + '/text_summary' in tensors, tag


def test_selfplay_logs_to_tensorboard(tmp_path):
    agent = SelfPlayPPO(device='cpu', channels=16, blocks=1)
    agent.learn(max_frames=4 * 8 * 8, num_envs=8, num_steps=8, batch_size=32, ppo_epochs=1, max_plies=30,
                snapshot_every=1, log_every=2, eval_games=4, eval_depths=(0, 1), plot_progress=False, seed=0,
                num_threads=2, tensorboard=tmp_path)
    events = read_events(tmp_path)
    scalars = events.Tags()['scalars']
    for tag in ('train/entropy', 'rollout/explained_variance', 'eval/vs_random', 'elo/vs_depth1', 'selfplay/mate',
                'opponents/frac_engine', 'schedule/shaping', 'pool/size', 'time/fps', 'norms/grad_tower'):
        assert tag in scalars, tag
    assert len(events.Scalars('eval/vs_random')) == 2
    assert events.Graph() and events.Tags()['images']
    assert any(tag.startswith('weights/tower/0/') for tag in events.Tags()['histograms'])


def test_missing_tensorboard_directory_means_no_logging(tmp_path):
    agent = SelfPlayPPO(device='cpu', channels=16, blocks=1)
    agent.learn(max_frames=64, num_envs=8, num_steps=8, batch_size=32, ppo_epochs=1, max_plies=30, eval_games=0,
                plot_progress=False, seed=0, num_threads=2)
    assert not list(tmp_path.iterdir())
