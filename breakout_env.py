from PPO.wrappers import make_atari, CropScreen, wrap_deepmind, wrap_pytorch
import matplotlib.pyplot as plt
import numpy as np
import gym
from gym import spaces


class BreakoutContinuous(gym.Wrapper):
    def __init__(self, env):
        super().__init__(self)
        self.observation_space = env.observation_space
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,))
        self.env = env

    def reset(self, **kwargs):
        self.last_obs = self.env.reset(**kwargs)
        return self.last_obs

    def step(self, action):
        paddle_pos = self.get_paddle_pos(self.last_obs[-1])
        target_pos = action * 41.5 + 41.5
        print(paddle_pos, target_pos)
        if paddle_pos > target_pos:
            print(' <- ')
            action = 3
        elif paddle_pos < target_pos:
            print(' -> ')
            action = 2
        else:
            print(' - ')
            action = 0
        self.last_obs, reward, done, info = self.env.step(action)
        return self.last_obs, reward, done, info

    def render(self, mode='human', **kwargs):
        return self.env.render(mode=mode, **kwargs)

    def get_paddle_pos(self, frame):
        pos = np.where(frame[83, :] > 0)[0]
        print(pos)
        if pos[0] > 0 and pos[-1] < 83:
            center = pos[0] + ((pos[-1] - pos[0]) / 2)
        elif pos[0] == 0:
            center = pos[-1] - 6
        elif pos[-1] == 83:
            center = pos[0] + 6
        return center


def make_env():
    env = make_atari('BreakoutNoFrameskip-v4')
    env = CropScreen(env, top=32, bottom=225, left=8, right=160)
    env = wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=True, scale=True)
    env = wrap_pytorch(env)
    env = BreakoutContinuous(env)
    return env


def plot_frames(obs):
    fig, axs = plt.subplots(1, 4)
    axs[0].imshow(obs[0], cmap='gray')
    axs[1].imshow(obs[1], cmap='gray')
    axs[2].imshow(obs[2], cmap='gray')
    axs[3].imshow(obs[3], cmap='gray')




env = make_env()
print('obs_space:', env.observation_space)
print('act_space:', env.action_space)
obs = env.reset()
for i in range(20):
    obs, r, d, _ = env.step(0.5)
plot_frames(obs)
