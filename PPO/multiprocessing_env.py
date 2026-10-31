#This code is from openai baseline
#https://github.com/openai/baselines/tree/master/baselines/common/vec_env

import traceback
import numpy as np
import multiprocessing as mp
from collections import OrderedDict
from PPO.util_vec_env import obs_space_info, copy_obs_dict, dict_to_obs

def worker(remote, parent_remote, env_fn_wrapper):
    """
    Every reply is a tuple (status, payload) with status 'ok' or 'error' (payload is the traceback),
    so that an exception in the environment is re-raised in the parent instead of hanging it.
    """
    parent_remote.close()
    try:
        env = env_fn_wrapper.x()
    except Exception:
        env = None
        init_error = traceback.format_exc()
    while True:
        cmd, data = remote.recv()
        if cmd == 'close':
            if env is not None:
                env.close()
            remote.close()
            break
        try:
            if env is None:
                raise RuntimeError('environment creation failed:\n' + init_error)
            if cmd == 'step':
                ob, reward, terminated, truncated, _ = env.step(data)
                info = {}
                if truncated and not terminated:
                    # the value of the cut-off state is needed to bootstrap (time limit is not a real terminal state)
                    info['final_observation'] = ob
                if terminated or truncated:
                    ob, _ = env.reset()
                remote.send(('ok', (ob, reward, terminated, truncated, info)))
            elif cmd == 'reset':
                ob, _ = env.reset()
                remote.send(('ok', ob))
            elif cmd == 'get_spaces':
                remote.send(('ok', (env.observation_space, env.action_space)))
            else:
                raise NotImplementedError(cmd)
        except Exception:
            remote.send(('error', traceback.format_exc()))

class VecEnv(object):
    """
    An abstract asynchronous, vectorized environment.
    """
    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a tuple of observation arrays.
        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.
        You should not call this if a step_async run is
        already pending.
        """
        pass

    def step_wait(self):
        """
        Wait for the step taken with step_async().
        Returns (obs, rews, terminateds, truncateds, infos):
         - obs: an array of observations, or a tuple of
                arrays of observations. Finished environments
                are automatically reset and return the first
                observation of the new episode.
         - rews: an array of rewards
         - terminateds: an array of "real end of episode" booleans
         - truncateds: an array of "time limit reached" booleans
         - infos: a sequence of dicts; an environment that was
                truncated (and not terminated) stores the last
                observation of the episode in 'final_observation'
        """
        pass

    def close(self):
        """
        Clean up the environments' resources.
        """
        pass

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    
class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

        
class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.nenvs = nenvs
        # spawn is used because forking a process that already runs torch threads can deadlock
        ctx = mp.get_context('spawn')
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(nenvs)])
        self.ps = [ctx.Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        try:
            self.remotes[0].send(('get_spaces', None))
            observation_space, action_space = self._receive(self.remotes[0])
        except Exception:
            self._terminate()
            raise
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    @staticmethod
    def _receive(remote):
        status, payload = remote.recv()
        if status == 'error':
            raise RuntimeError('Exception in environment worker:\n' + payload)
        return payload

    def _receive_all(self):
        # every pending reply is read, even after a failure, to keep the pipes consistent
        results, error = [], None
        for remote in self.remotes:
            try:
                results.append(self._receive(remote))
            except RuntimeError as e:
                error = error or e
        if error is not None:
            raise error
        return results

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        try:
            results = self._receive_all()
        finally:
            self.waiting = False
        obs, rews, terminateds, truncateds, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.array(terminateds), np.array(truncateds), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack(self._receive_all())

    def _terminate(self):
        for p in self.ps:
            if p.is_alive():
                p.terminate()
            p.join()
        self.closed = True

    def close(self):
        if self.closed:
            return
        try:
            if self.waiting:
                self._receive_all()
            for remote in self.remotes:
                remote.send(('close', None))
            for p in self.ps:
                p.join(timeout=10)
        except (RuntimeError, OSError, EOFError):
            pass
        self._terminate()
    def __len__(self):
        return self.nenvs

class DummyVecEnv(VecEnv):
    """
    Creates a simple vectorized wrapper for multiple environments, calling each environment in sequence on the current
    Python process. This is useful for computationally simple environment such as ``cartpole-v1``, as the overhead of
    multiprocess or multithread outweighs the environment computation time. This can also be used for RL methods that
    require a vectorized environment, but that you want a single environments to train with.

    :param env_fns: ([Gym Environment]) the list of environments to vectorize
    """

    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        obs_space = env.observation_space
        self.keys, shapes, dtypes = obs_space_info(obs_space)

        self.buf_obs = OrderedDict([
            (k, np.zeros((self.num_envs,) + tuple(shapes[k]), dtype=dtypes[k]))
            for k in self.keys])
        self.buf_terminateds = np.zeros((self.num_envs,), dtype=bool)
        self.buf_truncateds = np.zeros((self.num_envs,), dtype=bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None
        self.metadata = env.metadata

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        for env_idx in range(self.num_envs):
            obs, self.buf_rews[env_idx], terminated, truncated, _ = self.envs[env_idx].step(self.actions[env_idx])
            self.buf_terminateds[env_idx] = terminated
            self.buf_truncateds[env_idx] = truncated
            self.buf_infos[env_idx] = {}
            if truncated and not terminated:
                self.buf_infos[env_idx]['final_observation'] = obs
            if terminated or truncated:
                obs, _ = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_terminateds),
                np.copy(self.buf_truncateds), self.buf_infos.copy())

    def reset(self):
        for env_idx in range(self.num_envs):
            obs, _ = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
        return self._obs_from_buf()

    def close(self):
        for env in self.envs:
            env.close()

    def get_images(self):
        return [env.render() for env in self.envs]

    def render(self, *args, **kwargs):
        if self.num_envs == 1:
            return self.envs[0].render(*args, **kwargs)
        raise NotImplementedError('render is only supported with a single environment, use get_images instead')

    def _save_obs(self, env_idx, obs):
        for key in self.keys:
            if key is None:
                self.buf_obs[key][env_idx] = obs
            else:
                self.buf_obs[key][env_idx] = obs[key]

    def _obs_from_buf(self):
        return dict_to_obs(self.observation_space, copy_obs_dict(self.buf_obs))

    def get_attr(self, attr_name, indices=None):
        """Return attribute from vectorized environment (see base class)."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, attr_name) for env_i in target_envs]

    def set_attr(self, attr_name, value, indices=None):
        """Set attribute inside vectorized environments (see base class)."""
        target_envs = self._get_target_envs(indices)
        for env_i in target_envs:
            setattr(env_i, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Call instance methods of vectorized environments."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, method_name)(*method_args, **method_kwargs) for env_i in target_envs]

    def _get_indices(self, indices):
        if indices is None:
            return range(self.num_envs)
        if isinstance(indices, int):
            return [indices]
        return indices

    def _get_target_envs(self, indices):
        indices = self._get_indices(indices)
        return [self.envs[i] for i in indices]
