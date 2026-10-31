# Proximal Policy Optimization (PPO)

This is a simple but efficient implementation of the PPO algorithm by OpenAI. I have borrowed the code for the GAE computation and the PPO optimization from [RL-Adventure-2](https://github.com/higgsfield/RL-Adventure-2) and I thank the author for their work, which has been very important for my understanding of the algorithm. Also, I thank the maintainers of [stable-baselines](https://github.com/hill-a/stable-baselines); my implementation is inspired by their work in two ways:
1. Easy to use interface, in sklearn style.
2. Implementation of the actor-critic policy and feature extractor.

## Structure

- **PPO** - PPO (algorithm implementation)
  - **policy** (stochastic actor-critic policy with CNN or Mlp feature extractor)
  - **multiprocessing_env** (utility to parallelize training across the CPU cores)
  - **util_vec_env** (utility for multiprocessing_env)
  - **wrappers** (utility to wrap Atari games from Gym)

## Setup

The project uses [uv](https://docs.astral.sh/uv/) and [Gymnasium](https://gymnasium.farama.org/) (the maintained successor of OpenAI Gym; Atari games come from `ale-py`).

```bash
uv sync
```

Environments follow the Gymnasium API (`reset() -> (obs, info)`, `step() -> (obs, reward, terminated, truncated, info)`). To watch an agent inline in a notebook, create the environment with `render_mode='rgb_array'` and pass `vis=True` to `run_test_env` (frames are shown in the cell output, `fps` limits the playback speed).

## Use

### Import the necessary modules

```python
import gymnasium as gym
from PPO.PPO import PPO, run_test_env
from PPO.wrappers import make_atari, wrap_deepmind, wrap_pytorch
```

### Define the learning hyperparameters

```python
max_frames = int(1e7)                 # max number of frames for learning
max_rewards = None                    # if specified, learning will be interrupted when the agent reaches the reward in a test
num_steps = 128                       # number of steps to run for each environment before updating the policy
num_envs = -1                         # number of environments to run in parallel (-1 uses all the CPU cores available)
batch_size = 256                      # batch size for the policy update (batch_size = num_steps * num_envs / n_minibatches)
ppo_epochs = 4                        # number of iterations along the memory for each policy update
clip_range = 0.2                      # value for clipping the objective function ('lin_0.2' means the clip_range will decrease to 0 at max_frames linearly)
lr = 2.5e-4                           # learning rate for the policy optimization ('lin_2.5e-4' means the learning_rate will decrease to 0 at max_frames linearly)
vf_coef = 0.5                         # critic_loss weight in the objective function
ent_coef = 0.01                       # entropy weight in the objective function
gamma = 0.99                          # coefficient for general advantage estimation
lam = 0.95                            # lambda coefficient for general advantage estimation
clip_grad_norm = 0.5                  # max norm for gradient clipping
std_decay = False                     # if True, the policy standard deviation decays linearly to 0 at max_frames
clip_log_std = None                   # if specified, log_std is clipped to this maximum value after every update
log_every = 10                        # number of policy updates between each log: the policy is tested, a checkpoint is saved and the history is plotted
runs_for_episode = 1                  # number of test environment runs summed in each test episode
test_episodes = 5                     # number of episodes to run when testing the policy performance; the result is the mean between all the episode rewards
log_file = './logs/checkpoint.pth'    # checkpoint file name (contains all the data to restart the learning process or to use a trained agent)
device = 'auto'                       # device to use for evaluation and optimization; 'auto' will use the GPU if available
```

### Defining the learning and testing environments
The learning environment needs to be a callable that returns a function to make a wrapped environment

```python
def make_env(render_mode=None):
    # return a function that makes a wrapped environment
    def _thunk():
        env = make_atari('PongNoFrameskip-v4', render_mode=render_mode)
        env = wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=True, scale=True)
        env = wrap_pytorch(env)
        return env
    return _thunk
```
The testing environment is a simple environment that could be wrapped differently from the learning environment for better metrics
```python
def make_test_env(render_mode=None):
    # return a wrapped environment
    env = make_atari('PongNoFrameskip-v4', render_mode=render_mode)
    env = wrap_deepmind(env, episode_life=False, clip_rewards=False, frame_stack=True, scale=True)
    env = wrap_pytorch(env)
    return env
```
If you don't need a different environment for testing
```python
def make_test_env():
    return None
```

### Making the PPO object for learning and predicting    
```python
ppo = PPO(make_env, device=device)
```

### Loading a previous checkpoint to continue learning or to use a trained agent
```python
ppo.load(log_file)
```

### Starting the learning or continuing the learning if a checkpoint has been loaded
```python
ppo.learn(
    max_frames, max_rewards=max_rewards, num_steps=num_steps, num_envs=num_envs, batch_size=batch_size,
    ppo_epochs=ppo_epochs, clip_range=clip_range, lr=lr, vf_coef=vf_coef, ent_coef=ent_coef, gamma=gamma, lam=lam,
    clip_grad_norm=clip_grad_norm, std_decay=std_decay, clip_log_std=clip_log_std, log_every=log_every,
    log_file=log_file, test_env=make_test_env(), runs_for_episode=runs_for_episode, test_episodes=test_episodes
)
```

### Loading and testing a trained agent
```python
ppo = PPO(make_env, device='cpu')
ppo.load(log_file)

env = make_env(render_mode='rgb_array')()
# env = make_test_env(render_mode='rgb_array')
n_episode = 3

# if deterministic is True, the policy predictions will not be stochastic, which could improve performance
mean_reward = run_test_env(env=env, model=ppo, runs_for_episode=1, n_episodes=n_episode, vis=True, deterministic=False)
env.close()

print('Mean reward over %i episodes = %f' % (n_episode, mean_reward))
```