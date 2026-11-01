from PPO.policy import ActorCritic
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import multiprocessing as mp
from PPO.multiprocessing_env import SubprocVecEnv, DummyVecEnv
from IPython.display import clear_output, display
import numpy as np
from PIL import Image
from gymnasium.spaces import Box
import time
from collections import deque

from PPO.tensorboard_logger import TensorBoardLogger, explained_variance

# the curves shown together in the "Custom Scalars" tab of TensorBoard
TENSORBOARD_LAYOUT = {
    'PPO': {
        'reward': ['Multiline', ['eval/reward', 'rollout/ep_return_mean']],
        'losses': ['Multiline', ['train/loss', 'train/actor_loss', 'train/critic_loss']],
        'policy update': ['Multiline', ['train/approx_kl', 'train/clip_fraction']],
        'entropy': ['Multiline', ['train/entropy']],
        'critic quality': ['Multiline', ['rollout/explained_variance']],
        'gradients': ['Multiline', ['train/grad_norm', 'norms/global_grads']],
        'speed': ['Multiline', ['time/fps']],
    },
}


def set_device(device):
    if device == 'auto':
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def to_tensor(obs, device):
    """uint8 observations are kept as they are (the network scales them), everything else becomes float32."""
    obs = torch.as_tensor(obs, device=device)
    return obs if obs.dtype == torch.uint8 else obs.float()


def schedule(value, progress):
    """Resolve a hyper-parameter: a number is constant, 'lin_x' decays linearly from x to 0, 'const_x' is x."""
    if isinstance(value, str):
        mode, value = value.split('_')
        value = float(value)
        if mode == 'lin':
            value *= 1 - progress
    return value


@torch.no_grad()
def compute_gae(rewards, values, masks, next_value, gamma, lam):
    """
    GAE over a rollout. rewards, values and masks have shape (num_steps, num_envs),
    next_value (num_envs,) is the value of the state following the last step.
    Returns the advantages with shape (num_steps, num_envs).
    """
    num_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(next_value)
    for step in reversed(range(num_steps)):
        step_next_value = next_value if step == num_steps - 1 else values[step + 1]
        delta = rewards[step] + gamma * step_next_value * masks[step] - values[step]
        gae = delta + gamma * lam * masks[step] * gae
        advantages[step] = gae
    return advantages


def ppo_update(states, actions, old_log_probs, returns, advantages, net, optimizer, lr, batch_size, ppo_epochs,
               clip_range, vf_coef, ent_coef, clip_grad_norm, progress, std_decay, clip_log_std=None,
               action_masks=None, target_kl=None):
    """
    All the rollout tensors live on the same device as the network and have the batch as first dimension.
    action_masks: optional boolean tensor of the legal actions of every sample, passed to net.evaluate.
    target_kl: if specified, the update is stopped as soon as the approximate KL divergence from the policy that
    collected the data exceeds 1.5 * target_kl.
    """
    lr = schedule(lr, progress)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    clip_range = schedule(clip_range, progress)

    num_samples = states.shape[0]
    # loss, critic_loss, actor_loss, approx_kl, clip_fraction, entropy, grad_norm (before clipping)
    stats = torch.zeros(7, device=states.device)
    num_batches = 0
    stop = False
    for _ in range(ppo_epochs):
        permutation = torch.randperm(num_samples, device=states.device)
        for start in range(0, num_samples, batch_size):
            idx = permutation[start:start + batch_size]
            if action_masks is None:
                new_log_probs, entropy, value = net.evaluate(states[idx], actions[idx])
            else:
                new_log_probs, entropy, value = net.evaluate(states[idx], actions[idx], action_masks[idx])
            log_ratio = new_log_probs - old_log_probs[idx]
            ratio = log_ratio.exp()
            advantage = advantages[idx]
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantage
            actor_loss = - torch.min(surr1, surr2).mean()
            critic_loss = (returns[idx] - value).pow(2).mean()
            loss = vf_coef * critic_loss + actor_loss - ent_coef * entropy.mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), clip_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1) - log_ratio).mean()
                clip_fraction = ((ratio - 1).abs() > clip_range).float().mean()
                stats += torch.stack([loss, critic_loss, actor_loss, approx_kl, clip_fraction, entropy.mean(),
                                      grad_norm])
                stop = target_kl is not None and approx_kl.item() > 1.5 * target_kl
            num_batches += 1
            if stop:
                break
        if stop:
            break

    loss, critic_loss, actor_loss, approx_kl, clip_fraction, entropy, grad_norm = (stats / num_batches).tolist()
    # networks for discrete actions that have no std (e.g. the chess one) just report the defaults
    opt_info = {'lr': lr,
                'clip_range': clip_range,
                'vf_coef': vf_coef,
                'std_coef': net.std_coef.item() if hasattr(net, 'std_coef') else 1.0,
                'loss': loss,
                'critic_loss': critic_loss,
                'actor_loss': actor_loss,
                'approx_kl': approx_kl,
                'clip_fraction': clip_fraction,
                'entropy': entropy,
                'grad_norm': grad_norm,
                'num_batches': num_batches,
                'std': net.log_std.detach().exp().cpu() if hasattr(net, 'log_std') else torch.ones(1)}

    with torch.no_grad():
        # the std only exists for continuous actions
        if std_decay and not net.discrete:
            net.std_coef.fill_(max(1 - progress, 1e-3))
        if clip_log_std is not None and hasattr(net, 'log_std'):
            net.log_std.clamp_(max=clip_log_std)

    return opt_info


def checkpoint_history(checkpoint):
    lr = []
    clip_range = []
    vf_coef = []
    std_coef = []
    loss = []
    critic_loss = []
    actor_loss = []
    approx_kl = []
    clip_fraction = []
    std = []

    for record in checkpoint['opt_info']:
        lr.append(record['lr'])
        clip_range.append(record['clip_range'])
        vf_coef.append(record['vf_coef'])
        record_std_coef = record.get('std_coef', 1.0)
        std_coef.append(record_std_coef)
        loss.append(record['loss'])
        critic_loss.append(record['critic_loss'] * record['vf_coef'])
        actor_loss.append(record['actor_loss'])
        approx_kl.append(record.get('approx_kl', float('nan')))
        clip_fraction.append(record.get('clip_fraction', float('nan')))
        std.append(record['std'].mean().item() * record_std_coef if 'std' in record else float('nan'))

    return {'lr': lr,
            'clip_range': clip_range,
            'vf_coef': vf_coef,
            'std_coef': std_coef,
            'loss': loss,
            'critic_loss': critic_loss,
            'actor_loss': actor_loss,
            'approx_kl': approx_kl,
            'clip_fraction': clip_fraction,
            'std': std}


def plot(checkpoint, fps):
    data = checkpoint_history(checkpoint)
    num_frames = checkpoint['frames']
    frames, rewards = checkpoint['rewards']
    time_elapsed = checkpoint['time_elapsed']
    clear_output(True)
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
    display(fig)
    plt.close(fig)
    print('frames       : %i' % num_frames)
    print('time         : %i' % int(time_elapsed / 60))
    print('fps          : %i' % fps)
    print('reward       : %.2f' % rewards[-1])
    print('lr           : %.6f' % data['lr'][-1])
    print('loss         : %.3f' % data['loss'][-1])
    print('critic_loss  : %.3f' % data['critic_loss'][-1])
    print('actor_loss   : %.3f' % data['actor_loss'][-1])
    print('approx_kl    : %.5f' % data['approx_kl'][-1])
    print('clip_fraction: %.3f' % data['clip_fraction'][-1])
    print('std          : %.3f' % data['std'][-1])
    print('clip_range   : %.3f' % data['clip_range'][-1])
    print('vf_coef      : %.3f' % data['vf_coef'][-1])
    print('std_coef     : %.3f' % data['std_coef'][-1])


def show_frame(env, handle, fps):
    """Display the current frame of an env created with render_mode='rgb_array' inline in the notebook."""
    frame = env.render()
    if frame is None:
        raise ValueError("vis=True requires an environment created with render_mode='rgb_array'")
    image = Image.fromarray(frame)
    if handle is None:
        handle = display(image, display_id=True)
    else:
        handle.update(image)
    if fps:
        time.sleep(1 / fps)
    return handle


def run_test_env(env, model, runs_for_episode, n_episodes, vis=False, deterministic=False, fps=30):
    """Run test episodes. With vis=True the env (render_mode='rgb_array') is rendered inline in the notebook."""
    rewards = []
    handle = None
    for i in range(n_episodes):
        state, _ = env.reset()
        if vis: handle = show_frame(env, handle, fps)
        done = False
        total_reward = 0
        for run in range(runs_for_episode):
            while not done:
                action = model.predict(state, deterministic)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                state = next_state
                if vis:
                    handle = show_frame(env, handle, fps)
                total_reward += reward
            state, _ = env.reset()
            done = False
        rewards.append(total_reward)
    return np.mean(rewards)


# numpy types stored in checkpoints (the test rewards), needed to load them with weights_only=True
_CHECKPOINT_SAFE_GLOBALS = [np._core.multiarray.scalar, np.dtype, np.dtypes.Float64DType]


class PPO(object):

    def __init__(self, make_env, device='auto'):
        super(PPO, self).__init__()
        self.device = set_device(device)
        self.make_env = make_env

    def make_net(self):
        env = self.make_env()()
        try:
            if isinstance(env.action_space, Box):
                n_action = env.action_space.shape[0]
                discrete = False
                self.action_low, self.action_high = env.action_space.low, env.action_space.high
            else:
                n_action = env.action_space.n
                discrete = True
                self.action_low = self.action_high = None
            self.policy = ActorCritic(env.observation_space.shape, n_action, discrete).to(self.device)
        finally:
            env.close()

    def make_optimizer(self):
        # the learning rate is set at every update (see ppo_update)
        return optim.Adam(self.policy.parameters(), eps=1e-5)

    def to_env_action(self, action):
        action = action.cpu().numpy()
        if self.action_low is not None:
            # the unclipped action is the one used for the log-probabilities
            action = np.clip(action, self.action_low, self.action_high)
        return action

    def save(self, file):
        self.checkpoint = {'policy': self.policy.state_dict(),
                           'optimizer': self.optimizer.state_dict(),
                           'frames': self.frame_idx,
                           'rewards': self.test_rewards,
                           'time_elapsed': self.time_elapsed,
                           'opt_info': self.opt_info}
        if file is not None:
            torch.save(self.checkpoint, file)

    def load(self, file):
        self.make_net()
        with torch.serialization.safe_globals(_CHECKPOINT_SAFE_GLOBALS):
            self.checkpoint = torch.load(file, map_location=self.device, weights_only=True)

        # checkpoints saved by older versions may lack some parameters (log_std, std_coef)
        policy_state = self.checkpoint['policy']
        missing = [k for k in self.policy.state_dict() if k not in policy_state]
        if missing:
            policy_state = {**self.policy.state_dict(), **policy_state}
        self.policy.load_state_dict(policy_state)

        self.optimizer = self.make_optimizer()
        if not missing:
            self.optimizer.load_state_dict(self.checkpoint['optimizer'])
            for param_group in self.optimizer.param_groups:
                param_group['eps'] = 1e-5
        self.frame_idx = self.checkpoint['frames']
        self.test_rewards = self.checkpoint['rewards']
        self.prev_time_elapsed = self.checkpoint['time_elapsed']
        self.opt_info = list(self.checkpoint.get('opt_info', []))

        # keep opt_info aligned with the test rewards history used by plot()
        n_missing = len(self.test_rewards[1]) - len(self.opt_info)
        if n_missing > 0:
            nan = float('nan')
            empty_record = {'lr': nan, 'clip_range': nan, 'vf_coef': nan, 'std_coef': nan, 'loss': nan,
                            'critic_loss': nan, 'actor_loss': nan, 'std': torch.full((1,), nan)}
            self.opt_info = [dict(empty_record) for _ in range(n_missing)] + self.opt_info

    def predict(self, obs, deterministic=False):
        obs = np.asarray(obs)
        single = obs.ndim == 3 or obs.ndim == 1
        if single:
            obs = np.expand_dims(obs, 0)
        with torch.no_grad():
            action, _, _ = self.policy.act(to_tensor(obs, self.device), deterministic)
        action = self.to_env_action(action)
        return action[0] if single else action

    def log_progress(self, eval_env, runs_for_episode, test_episodes, opt_info, fps, start_learning, log_file,
                     tb=None):
        test_reward = run_test_env(eval_env, self, runs_for_episode, test_episodes)
        self.test_rewards[0].append(self.frame_idx)
        self.test_rewards[1].append(float(test_reward))
        self.time_elapsed = time.time() - start_learning + self.prev_time_elapsed
        self.opt_info.append(opt_info)
        self.save(log_file)
        if tb is not None:
            tb.scalars(self.frame_idx, {'reward': test_reward}, 'eval/')
            tb.scalars(self.frame_idx, {'fps': fps, 'minutes': self.time_elapsed / 60}, 'time/')
            tb.weights(self.frame_idx, self.policy)
            tb.flush()
        plot(self.checkpoint, fps)
        return test_reward

    def learn(self, max_frames, max_rewards=None, num_envs=-1, lr=2.5e-4, num_steps=128, batch_size=256, ppo_epochs=4,
              clip_range=0.2, vf_coef=0.5, ent_coef=0.01, gamma=0.99, lam=0.95, clip_grad_norm=0.5, std_decay=False,
              clip_log_std=None, log_every=10, log_file=None, test_env=None, runs_for_episode=1, test_episodes=5,
              vec_env='auto', tensorboard=None):
        """
        vec_env: 'subproc' steps every environment in its own process, 'dummy' steps them sequentially in this
        process (faster for cheap environments); 'auto' uses 'dummy' for vector observations and 'subproc' for images.
        tensorboard: directory of the TensorBoard logs (or a TensorBoardLogger), in addition to the notebook plots.
        The curves use the number of frames as step, so resuming a run with the same directory continues them.
        """
        hparams = {k: v for k, v in locals().items() if k not in ('self', 'log_file', 'test_env', 'tensorboard')}

        if num_envs == -1:
            num_envs = mp.cpu_count()
        hparams['num_envs'] = num_envs
        env = self.make_env()()
        if vec_env == 'auto':
            vec_env = 'dummy' if len(env.observation_space.shape) == 1 else 'subproc'
        hparams['vec_env'] = vec_env
        envs = (DummyVecEnv if vec_env == 'dummy' else SubprocVecEnv)([self.make_env() for _ in range(num_envs)])
        eval_env = env if test_env is None else test_env
        tb = None

        try:
            tb = TensorBoardLogger.resolve(tensorboard)
            if not hasattr(self, 'policy'):
                self.make_net()
            if not hasattr(self, 'optimizer'):
                self.optimizer = self.make_optimizer()
            if not hasattr(self, 'frame_idx'):
                self.frame_idx = 0
            if not hasattr(self, 'test_rewards'):
                self.test_rewards = ([], [])
            if not hasattr(self, 'prev_time_elapsed'):
                self.prev_time_elapsed = 0
            if not hasattr(self, 'opt_info'):
                self.opt_info = []
            if tb is not None:
                sample_obs, _ = env.reset()
                tb.log_setup(self.policy, to_tensor(np.asarray(sample_obs)[None], self.device), hparams,
                             metrics=['eval/reward'], layout=TENSORBOARD_LAYOUT,
                             run_info={'observation space': env.observation_space, 'action space': env.action_space,
                                       'resumed at frame': self.frame_idx})
                tb.log_observation(sample_obs)

            # the rollout is stored on the device, so that the update never moves data across it
            device = self.device
            obs_shape = env.observation_space.shape
            obs_dtype = torch.uint8 if env.observation_space.dtype == np.uint8 else torch.float32
            if self.policy.discrete:
                action_shape, action_dtype = (), torch.long
            else:
                action_shape, action_dtype = (self.action_low.shape[0],), torch.float32
            obs_buf = torch.empty((num_steps, num_envs, *obs_shape), dtype=obs_dtype, device=device)
            action_buf = torch.empty((num_steps, num_envs, *action_shape), dtype=action_dtype, device=device)
            log_prob_buf = torch.empty((num_steps, num_envs), device=device)
            value_buf = torch.empty((num_steps, num_envs), device=device)
            reward_buf = np.zeros((num_steps, num_envs), dtype=np.float32)
            mask_buf = np.zeros((num_steps, num_envs), dtype=np.float32)

            ppo_cycles = 0
            logged_cycle = 0
            target_reached = False
            start_learning = time.time()
            start_cycle = time.time()
            start_cycle_frames = self.frame_idx
            episode_return = np.zeros(num_envs)
            episode_length = np.zeros(num_envs, dtype=np.int64)
            finished_episodes = deque(maxlen=100)  # (return, length) of the latest episodes
            state = envs.reset()
            while True:

                with torch.no_grad():
                    for step in range(num_steps):
                        state_t = to_tensor(state, device)
                        action, log_prob, value = self.policy.act(state_t)
                        next_state, reward, terminated, truncated, infos = envs.step(self.to_env_action(action))

                        obs_buf[step] = state_t
                        action_buf[step] = action
                        log_prob_buf[step] = log_prob
                        value_buf[step] = value
                        reward_buf[step] = reward
                        mask_buf[step] = 1 - (terminated | truncated)

                        # a time limit is not a real terminal state: bootstrap with the value of the last state
                        cut_off = np.flatnonzero(truncated & ~terminated)
                        if len(cut_off) > 0:
                            final_obs = np.stack([infos[i]['final_observation'] for i in cut_off])
                            final_value = self.policy.get_value(to_tensor(final_obs, device)).cpu().numpy()
                            reward_buf[step, cut_off] += gamma * final_value

                        episode_return += reward
                        episode_length += 1
                        for i in np.flatnonzero(terminated | truncated):
                            finished_episodes.append((episode_return[i], episode_length[i]))
                            episode_return[i] = episode_length[i] = 0

                        state = next_state
                        self.frame_idx += num_envs

                    next_value = self.policy.get_value(to_tensor(state, device))
                    rewards = torch.as_tensor(reward_buf, device=device)
                    masks = torch.as_tensor(mask_buf, device=device)
                    advantages = compute_gae(rewards, value_buf, masks, next_value, gamma, lam)
                    returns = advantages + value_buf
                    advantages = (advantages - advantages.mean()) / (advantages.std(correction=0) + 1e-8)

                _opt_info = ppo_update(obs_buf.flatten(0, 1), action_buf.flatten(0, 1), log_prob_buf.flatten(),
                                       returns.flatten(), advantages.flatten(), self.policy, self.optimizer, lr,
                                       batch_size, ppo_epochs, clip_range, vf_coef, ent_coef, clip_grad_norm,
                                       progress=min(self.frame_idx / max_frames, 1.0), std_decay=std_decay,
                                       clip_log_std=clip_log_std)
                ppo_cycles += 1
                if tb is not None:
                    progress = min(self.frame_idx / max_frames, 1.0)
                    tb.update(self.frame_idx, _opt_info, log_std=not self.policy.discrete)
                    tb.scalars(self.frame_idx, {'progress': progress}, 'schedule/')
                    tb.scalars(self.frame_idx, {
                        'explained_variance': explained_variance(value_buf.flatten(), returns.flatten()),
                        'value_mean': value_buf.mean().item(), 'return_mean': returns.mean().item(),
                        'reward_mean': reward_buf.mean(),
                        **({'ep_return_mean': np.mean([r for r, _ in finished_episodes]),
                            'ep_length_mean': np.mean([n for _, n in finished_episodes])}
                           if finished_episodes else {})}, 'rollout/')
                if ppo_cycles % log_every == 0:
                    fps = int((self.frame_idx - start_cycle_frames) / (time.time() - start_cycle))
                    if tb is not None:
                        n_actions = self.policy.actor.out_features
                        tb.histogram(self.frame_idx, 'rollout/actions', action_buf,
                                     bins=np.arange(n_actions + 1) - 0.5 if self.policy.discrete else 'tensorflow')
                        tb.histogram(self.frame_idx, 'rollout/values', value_buf)
                        tb.histogram(self.frame_idx, 'rollout/returns', returns)
                    test_reward = self.log_progress(eval_env, runs_for_episode, test_episodes, _opt_info, fps,
                                                    start_learning, log_file, tb)
                    logged_cycle = ppo_cycles
                    start_cycle = time.time()
                    start_cycle_frames = self.frame_idx
                    if max_rewards is not None and test_reward >= max_rewards:
                        target_reached = True
                        break
                if self.frame_idx >= max_frames:
                    break

            if target_reached:
                print('> Max reward target reached!')
            elif logged_cycle != ppo_cycles:
                fps = int((self.frame_idx - start_cycle_frames) / (time.time() - start_cycle))
                self.log_progress(eval_env, runs_for_episode, test_episodes, _opt_info, fps, start_learning, log_file,
                                  tb)
        finally:
            if tb is not None and tb is not tensorboard:
                tb.close()  # only the loggers created here are closed
            envs.close()
            env.close()
