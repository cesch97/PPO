from PPO.policy import ActorCritic
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import multiprocessing as mp
from PPO.multiprocessing_env import SubprocVecEnv
from IPython.display import clear_output
import numpy as np
from gym.spaces import Box
import time


def set_device(device):
    if device == 'auto':
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == 'cpu':
        return torch.device("cpu")


def compute_gae(next_value, rewards, masks, values, gamma, lam):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * lam * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns


class PPO_Dataset(Dataset):
    def __init__(self, len):
        self.states = None
        self.actions = None
        self.log_probs = None
        self.returns = None
        self.advantage = None
        self.len = len

    def load_data(self, states, actions, log_probs, returns, advantage):
        self.states = states
        self.actions = actions
        self.log_probs = log_probs
        self.returns = returns
        self.advantage = advantage

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.log_probs[idx], self.returns[idx], self.advantage[idx]


def ppo_update(dataloader, net, device, optimizer, lr, ppo_epochs, clip_range,
               vf_coef, ent_coef, clip_grad_norm, progress, std_decay, clip_log_std=None):

    if isinstance(lr, str):
        mode, lr = lr.split('_')
        lr = float(lr)
        if mode == 'lin':
            lr *= 1 - progress
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    if isinstance(clip_range, str):
        mode, clip_range = clip_range.split('_')
        clip_range = float(clip_range)
        if mode == 'lin':
            clip_range *= 1 - progress

    for _ in range(ppo_epochs):
        for state, action, old_log_probs, returns, advantage in dataloader:
            optimizer.zero_grad()
            dist, value = net(state.to(device))
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action.to(device))
            ratio = (new_log_probs - old_log_probs.to(device)).exp()
            advantage = advantage.to(device)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantage
            actor_loss = - torch.min(surr1, surr2).mean()
            critic_loss = (returns.to(device) - value).pow(2).mean()
            loss = vf_coef * critic_loss + actor_loss - ent_coef * entropy
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip_grad_norm)
            optimizer.step()

    state_dict = net.state_dict()
    opt_info = {'lr': lr,
                'clip_range': clip_range,
                'vf_coef': vf_coef,
                'std_coef': state_dict['std_coef'].item(),
                'loss': loss.item(),
                'critic_loss': critic_loss.item(),
                'actor_loss': actor_loss.item(),
                'std': state_dict['log_std'].exp().cpu()}

    if std_decay:
        state_dict['std_coef'] = torch.tensor(1 - progress).to(device)
    if clip_log_std is not None:
        state_dict['log_std'] = torch.clamp(state_dict['log_std'], max=clip_log_std)
    net.load_state_dict(state_dict)

    return opt_info


def checkpoint_history(checkpoint):
    lr = []
    clip_range = []
    vf_coef = []
    std_coef = []
    loss = []
    critic_loss = []
    actor_loss = []
    std = []

    for record in checkpoint['opt_info']:
        lr.append(record['lr'])
        clip_range.append(record['clip_range'])
        vf_coef.append(record['vf_coef'])
        std_coef.append(record['std_coef'])
        loss.append(record['loss'])
        critic_loss.append(record['critic_loss'] * record['vf_coef'])
        actor_loss.append(record['actor_loss'])
        std.append(record['std'].mean().item() * record['std_coef'])

    return {'lr': lr,
            'clip_range': clip_range,
            'vf_coef': vf_coef,
            'std_coef': std_coef,
            'loss': loss,
            'critic_loss': critic_loss,
            'actor_loss': actor_loss,
            'std': std}


def plot(checkpoint, fps):
    data = checkpoint_history(checkpoint)
    num_frames = checkpoint['frames']
    frames, rewards = checkpoint['rewards']
    time_elapsed = checkpoint['time_elapsed']
    clear_output(True)
    plt.clf()
    fig = plt.figure(constrained_layout=True)
    spec = fig.add_gridspec(ncols=1, nrows=3, height_ratios=[2, 1, 1], hspace=0)
    ax_reward = fig.add_subplot(spec[0, 0])
    ax_reward.plot(frames, rewards)
    ax_loss = fig.add_subplot(spec[1, 0])
    ax_loss.plot(frames, data['loss'])
    ax_loss.plot(frames, data['critic_loss'])
    ax_loss.plot(frames, data['actor_loss'])
    ax_std = fig.add_subplot(spec[2, 0])
    ax_std.plot(frames, data['std'])
    plt.pause(0.01)
    print('frames     : %i' % num_frames)
    print('time       : %i' % int(time_elapsed / 60))
    print('fps        : %i' % fps)
    print('reward     : %.2f' % rewards[-1])
    print('lr         : %.6f' % data['lr'][-1])
    print('loss       : %.3f' % data['loss'][-1])
    print('critic_loss: %.3f' % data['critic_loss'][-1])
    print('actor_loss : %.3f' % data['actor_loss'][-1])
    print('std        : %.3f' % data['std'][-1])
    print('clip_range : %.3f' % data['clip_range'][-1])
    print('vf_coef    : %.3f' % data['vf_coef'][-1])
    print('std_coef   : %.3f' % data['std_coef'][-1])


def run_test_env(env, model, runs_for_episode, n_episodes, vis=False, deterministic=False):
    rewards = []
    for i in range(n_episodes):
        state = env.reset()
        if vis: env.render()
        done = False
        total_reward = 0
        for run in range(runs_for_episode):
            while not done:
                action = model.predict(state, deterministic)
                next_state, reward, done, _ = env.step(action)
                state = next_state
                if vis:
                    env.render()
                total_reward += reward
            state = env.reset()
            done = False
        rewards.append(total_reward)
    return np.mean(rewards)


class PPO(object):

    def __init__(self, make_env, device='auto'):
        super(PPO, self).__init__()
        self.device = set_device(device)
        self.make_env = make_env

    def make_net(self):
        env = self.make_env()()
        if isinstance(env.action_space, Box):
            n_action = env.action_space.shape[0]
            discrete = False
        else:
            n_action = env.action_space.n
            discrete = True
        self.policy = ActorCritic(env.observation_space.shape, n_action, discrete).to(self.device)

    def save(self, file):
        self.checkpoint = {'policy': self.policy.state_dict(),
                           'optimizer': self.optimizer.state_dict(),
                           'frames': self.frame_idx,
                           'rewards': self.test_rewards,
                           'time_elapsed': self.time_elapsed,
                           'opt_info': self.opt_info}
        torch.save(self.checkpoint, file)

    def load(self, file):
        self.make_net()
        self.checkpoint = torch.load(file)
        self.policy.load_state_dict(self.checkpoint['policy'])
        self.optimizer = optim.Adam(self.policy.parameters())
        self.optimizer.load_state_dict(self.checkpoint['optimizer'])
        self.frame_idx = self.checkpoint['frames']
        self.test_rewards = self.checkpoint['rewards']
        self.prev_time_elapsed = self.checkpoint['time_elapsed']
        self.opt_info = self.checkpoint['opt_info']

    def predict(self, obs, deterministic=False):
        old_shape = obs.shape
        if len(old_shape) == 3 or len(old_shape) == 1:
            obs = np.expand_dims(obs, 0)
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            _, (_, action) = self.policy(obs, action=True, deterministic=deterministic)
        if len(old_shape) == 3 or len(old_shape) == 1:
            return action[0]
        return action

    def learn(self, max_frames, max_rewards=None, num_envs=-1, lr=2.5e-4, num_steps=128, batch_size=256, ppo_epochs=4,
              clip_range=0.2, vf_coef=0.5, ent_coef=0.01, gamma=0.99, lam=0.95, clip_grad_norm=0.5, std_decay=False, clip_log_std=None,
              log_every=10, log_file=None, test_env=None, runs_for_episode=1, test_episodes=5):

        if num_envs == -1:
            num_envs = mp.cpu_count()
        envs = SubprocVecEnv([self.make_env() for i in range(num_envs)])
        env = self.make_env()()

        if not hasattr(self, 'policy'):
            self.make_net()
        if not hasattr(self, 'optimizer'):
            if isinstance(lr, str):
                self.optimizer = optim.Adam(self.policy.parameters())
            else:
                self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        if not hasattr(self, 'frame_idx'):
            self.frame_idx = 0
        if not hasattr(self, 'test_rewards'):
            self.test_rewards = ([], [])
        if not hasattr(self, 'prev_time_elapsed'):
            self.prev_time_elapsed = 0
        if not hasattr(self, 'opt_info'):
            self.opt_info = []


        self.ppo_dataset = PPO_Dataset(envs.num_envs * num_steps)
        self.ppo_dataloader = DataLoader(self.ppo_dataset, batch_size, shuffle=True)

        ppo_cycles = 0
        start_learning = time.time()
        start_cycle = time.time()
        state = envs.reset()
        while True:

            log_probs = []
            values = []
            states = []
            actions = []
            rewards = []
            masks = []
            entropy = 0

            with torch.no_grad():
                for _ in range(num_steps):
                    state = torch.FloatTensor(state).to(self.device)
                    (dist, value), (sample, action) = self.policy(state, action=True)
                    value = value.cpu()
                    next_state, reward, done, _ = envs.step(action)
                    log_prob = dist.log_prob(sample)
                    entropy += dist.entropy().mean()
                    log_probs.append(log_prob)
                    values.append(value)
                    rewards.append(torch.FloatTensor(reward).unsqueeze(1))
                    masks.append(torch.FloatTensor(1 - done).unsqueeze(1))
                    states.append(state.cpu())
                    actions.append(sample)
                    state = next_state
                    self.frame_idx += num_envs

                next_state = torch.FloatTensor(next_state).to(self.device)
                _, next_value = self.policy(next_state)
                returns = compute_gae(next_value.cpu(), rewards, masks, values, gamma, lam)
                returns = torch.cat(returns)
                log_probs = torch.cat(log_probs)
                values = torch.cat(values)
                states = torch.cat(states)
                actions = torch.cat(actions)
                advantage = returns - values

            self.ppo_dataset.load_data(states, actions, log_probs, returns, advantage)
            _opt_info = ppo_update(self.ppo_dataloader, self.policy, self.device, self.optimizer, lr, ppo_epochs,
                                   clip_range, vf_coef, ent_coef, clip_grad_norm,
                                   progress=self.frame_idx/max_frames, std_decay=std_decay, clip_log_std=clip_log_std)
            ppo_cycles += 1
            if ppo_cycles % log_every == log_every - 1:
                if test_env is None:
                    test_reward = run_test_env(env, self, runs_for_episode, test_episodes)
                else:
                    test_reward = run_test_env(test_env, self, runs_for_episode, test_episodes)
                self.test_rewards[0].append(self.frame_idx)
                self.test_rewards[1].append(test_reward)
                self.time_elapsed = time.time() - start_learning + self.prev_time_elapsed
                fps = int((num_steps * num_envs * log_every) / (time.time() - start_cycle))
                start_cycle = time.time()
                self.opt_info.append(_opt_info)
                self.save(log_file)
                plot(self.checkpoint, fps)
                if max_rewards is not None and 'test_reward' in locals():
                    if test_reward >= max_rewards:
                        break
            if self.frame_idx >= max_frames:
                break
        if max_rewards is not None and 'test_reward' in locals():
            if test_reward >= max_rewards:
                print('> Max reward target reached!')
        else:
            if test_env is None:
                test_reward = run_test_env(env, self, runs_for_episode, test_episodes)
            else:
                test_reward = run_test_env(test_env, self, runs_for_episode, test_episodes)
            self.test_rewards[0].append(self.frame_idx)
            self.test_rewards[1].append(test_reward)
            self.time_elapsed = time.time() - start_learning + self.prev_time_elapsed
            fps = int((num_steps * num_envs * log_every) / (time.time() - start_cycle))
            self.opt_info.append(_opt_info)
            self.save(log_file)
            plot(self.checkpoint, fps)



































