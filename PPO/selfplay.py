"""
Self-play PPO for chess.

How the data is generated (every environment of the batch plays its own game, all of them are stepped together):
 - one network plays both colours: observations and actions are always from the point of view of the player to move;
 - negamax GAE: the game is zero-sum, so the value of a position is the negated value of the next one;
 - only legal moves can be sampled (action mask stored in the rollout, so the PPO ratio is exact);
 - every game gets an opponent when it starts: the current network (plain self-play), a past snapshot of it taken from
   an opponent pool (fictitious self-play, prioritised on the snapshots that are harder to beat: avoids strategy
   cycling and forgetting) or a built-in baseline engine (random player / shallow alpha-beta) that provides real
   checkmates while the network is still weak. The learned colour is random, only the learner's moves are trained on;
 - games start from a few random half-moves (diverse positions) and are cut at `max_plies`: a cut game is bootstrapped
   with the value of the last position (it is a time limit, not a result);
 - reward: +1 to the player who delivers checkmate, 0 for draws, plus a potential-based material shaping
   gamma * Phi(s') - Phi(s) with Phi the material balance: it speeds up the early learning without changing the optimal
   policy and it is annealed to 0 during training.
"""
import copy
import time
from collections import Counter, defaultdict, deque

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from IPython.display import clear_output, display

from PPO.PPO import ppo_update, schedule, set_device
from PPO.chess_env import CHECKMATE, MAXPLIES, NUM_ACTIONS, NUM_PLANES, OUTCOME_NAMES, ChessVecEnv, encode_move
from PPO.chess_policy import ChessNet

SELF, POOL, ENGINE = 0, 1, 2
MATERIAL_SCALE = 39.0  # material of one side at the start (in pawns)


@torch.no_grad()
def negamax_gae(rewards, values, masks, next_value, gamma, lam):
    """
    GAE for a zero-sum game where the players alternate at every step. rewards, values and masks have shape
    (num_steps, num_envs): reward and value are always from the point of view of the player to move at that step, so
    the next state counts with a negative sign. next_value (num_envs,) is the value of the state after the last step.
    """
    num_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(next_value)
    for step in reversed(range(num_steps)):
        step_next_value = next_value if step == num_steps - 1 else values[step + 1]
        delta = rewards[step] - gamma * step_next_value * masks[step] - values[step]
        gae = delta - gamma * lam * masks[step] * gae
        advantages[step] = gae
    return advantages


# --------------------------------------------------------------------------------------------------------- players
class NetPlayer:
    """Plays the legal move sampled from a ChessNet (deterministic=True plays the most likely move)."""

    def __init__(self, policy, device=None, deterministic=False, temperature=1.0):
        self.policy = policy
        self.device = device if device is not None else next(policy.parameters()).device
        self.deterministic = deterministic
        self.temperature = temperature

    @torch.no_grad()
    def act(self, env, idx):
        obs = torch.from_numpy(env.obs[idx]).to(self.device)
        mask = torch.from_numpy(env.mask[idx]).to(self.device)
        logits = self.policy(obs, mask)[0].logits
        if self.deterministic:
            action = logits.argmax(-1)
        else:
            action = torch.distributions.Categorical(logits=logits / self.temperature, validate_args=False).sample()
        return action.cpu().numpy().astype(np.int32)


class EnginePlayer:
    """The built-in baseline: depth 0 plays random legal moves, depth d >= 1 is an alpha-beta search of depth d."""

    def __init__(self, depth=0, noise=10):
        self.depth = depth
        self.noise = noise

    def act(self, env, idx):
        return env.engine_actions(idx, self.depth, self.noise)[idx].copy()


def elo_difference(score):
    """Elo difference of player A over B that corresponds to an average score (1 = win, 0.5 = draw)."""
    score = min(max(score, 0.01), 0.99)
    return 400 * np.log10(score / (1 - score))


def action_to_move(env, i, action):
    """(from_sq, to_sq, promotion) of the legal move of env i that has the given action index."""
    side = int(env.side[i])
    for move in env.legal_moves(i):
        if encode_move(side, *move) == action:
            return move
    raise ValueError('action %i is not legal' % action)


def play_match(player_a, player_b, n_games=64, max_plies=300, open_plies=(0, 6), seed=None, record_game=False):
    """
    Plays `n_games` games between two players in parallel, `player_a` has white in the even ones.
    Returns a dict with wins/draws/losses of player A, its mean score, the end reasons and (record_game=True) the list
    of (from_sq, to_sq, promotion) moves of the first game.
    """
    env = ChessVecEnv(n_games, max_plies=max_plies, open_plies=open_plies, seed=seed)
    try:
        env.reset()
        a_color = (np.arange(n_games) % 2).astype(np.int8)  # colour of player A: 0 white, 1 black
        score = np.full(n_games, np.nan)
        reasons = [None] * n_games
        moves = []
        while np.isnan(score).any():  # finished games restart, their next result is ignored
            side = env.side.copy()
            a_turn = side == a_color
            actions = np.zeros(n_games, dtype=np.int32)
            for player, idx in ((player_a, np.flatnonzero(a_turn)), (player_b, np.flatnonzero(~a_turn))):
                if len(idx):
                    actions[idx] = player.act(env, idx)
            if record_game and np.isnan(score[0]):
                moves.append(action_to_move(env, 0, actions[0]))
            _, _, info = env.step(actions)
            for i in np.flatnonzero(info.done & np.isnan(score)):
                reasons[i] = OUTCOME_NAMES[info.outcome[i]]
                score[i] = (1.0 if side[i] == a_color[i] else 0.0) if info.outcome[i] == CHECKMATE else 0.5
    finally:
        env.close()
    result = {'wins': int((score == 1).sum()), 'draws': int((score == 0.5).sum()), 'losses': int((score == 0).sum()),
              'score': float(score.mean()), 'reasons': dict(Counter(reasons))}
    if record_game:
        result['moves'] = moves
    return result


def evaluate_policy(policy, device, engine_depths=(0, 1, 2), n_games=32, max_plies=200, seed=None):
    """Score (1 = always wins) of the network against the baseline engines, with its stochastic policy."""
    player = NetPlayer(policy, device)
    results = {}
    for depth in engine_depths:
        name = 'random' if depth == 0 else 'depth%i' % depth
        results[name] = play_match(player, EnginePlayer(depth), n_games, max_plies, seed=seed)['score']
    return results


# --------------------------------------------------------------------------------------------------- opponent pool
class OpponentPool:
    """
    Frozen snapshots of the learner. When the pool is full a random old snapshot is dropped (the pool keeps a spread of
    the past, not only the latest versions) and snapshots are sampled with a probability that grows when the learner
    scores less against them (prioritised fictitious self-play).
    """

    def __init__(self, max_size=8, priority=2.0, ema=0.9):
        self.max_size = max_size
        self.priority = priority
        self.ema = ema
        self.items = {}  # uid -> dict(net, score, frames)
        self._next_uid = 0

    def __len__(self):
        return len(self.items)

    def _insert(self, net, score, frames):
        self.items[self._next_uid] = {'net': net.eval().requires_grad_(False), 'score': score, 'frames': frames}
        self._next_uid += 1

    def add(self, policy, frames, score=0.5):
        while len(self.items) >= self.max_size:
            victims = sorted(self.items)[:-1] or list(self.items)  # the newest snapshot is kept
            del self.items[victims[np.random.randint(len(victims))]]
        self._insert(copy.deepcopy(policy), score, frames)

    def sample(self, rng, n):
        uids = np.array(sorted(self.items))
        weights = np.array([(1.0 - self.items[u]['score'] + 0.05) ** self.priority for u in uids])
        return rng.choice(uids, size=n, p=weights / weights.sum())

    def update(self, uid, learner_score):
        if uid in self.items:
            item = self.items[uid]
            item['score'] = self.ema * item['score'] + (1 - self.ema) * learner_score

    def state_dict(self):
        return [{'state': item['net'].state_dict(), 'score': float(item['score']), 'frames': int(item['frames'])}
                for _, item in sorted(self.items.items())]

    def load_state_dict(self, states, make_net):
        self.items.clear()
        for state in states:
            net = make_net()
            net.load_state_dict(state['state'])
            self._insert(net, state['score'], state['frames'])


# --------------------------------------------------------------------------------------------------------- trainer
class SelfPlayPPO:

    def __init__(self, device='auto', channels=64, blocks=6, amp=False):
        """amp=True runs the network in bfloat16 on the GPU (faster, slightly less precise)."""
        self.device = set_device(device)
        self.policy = ChessNet(channels, blocks, amp).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), eps=1e-5)
        self.pool = OpponentPool()
        self.frame_idx = 0
        self.time_elapsed = 0.0
        self.history = []

    # ------------------------------------------------------------------------------------------ checkpoints
    def save(self, file):
        checkpoint = {'policy': self.policy.state_dict(), 'optimizer': self.optimizer.state_dict(),
                      'config': {**self.policy.config, 'amp': self.policy.amp}, 'frames': self.frame_idx,
                      'time_elapsed': self.time_elapsed, 'history': self.history, 'pool': self.pool.state_dict()}
        torch.save(checkpoint, file)

    @classmethod
    def load(cls, file, device='auto'):
        checkpoint = torch.load(file, map_location=set_device(device), weights_only=True)
        agent = cls(device, **checkpoint['config'])
        agent.policy.load_state_dict(checkpoint['policy'])
        agent.optimizer.load_state_dict(checkpoint['optimizer'])
        agent.frame_idx = checkpoint['frames']
        agent.time_elapsed = checkpoint['time_elapsed']
        agent.history = checkpoint['history']
        agent.pool.load_state_dict(checkpoint['pool'], agent._make_net)
        return agent

    def _make_net(self):
        return ChessNet(self.policy.config['channels'], self.policy.config['blocks'], self.policy.amp).to(self.device)

    def player(self, deterministic=False, temperature=1.0):
        return NetPlayer(self.policy, self.device, deterministic, temperature)

    # ------------------------------------------------------------------------------------------ training
    def learn(self, max_frames, num_envs=256, num_steps=64, batch_size=1024, ppo_epochs=4, lr=3e-4, clip_range=0.2,
              vf_coef=0.5, ent_coef=0.01, gamma=0.99, lam=0.95, clip_grad_norm=0.5, target_kl=None,
              max_plies=200, open_plies=(0, 8), shaping='lin_1.0', mate_shaping='lin_0.5', draw_penalty='lin_0.5',
              p_self=0.5, p_pool=0.25, p_engine='lin_0.6', engine_depths=(0, 1, 2), engine_noise=30,
              snapshot_every=10, pool_size=8, log_every=10, log_file=None, eval_games=32, eval_depths=(0, 1, 2),
              plot_progress=True, num_threads=None, seed=None):
        """
        max_frames: total number of half-moves played by all the environments (counting the opponents' ones).
        num_steps: half-moves played by every environment between two updates (the batch has num_envs * num_steps).
        lr, clip_range, ent_coef, shaping: a number or 'lin_x' (decays linearly from x to 0 at max_frames).
        shaping / mate_shaping: weights of the two potential-based shaping rewards (they do not change the optimal
            policy): the material balance, and the mating technique (weaker king pushed to the edge, kings close).
        draw_penalty: weight of the penalty for drawing (stalemate, repetition, ...) while ahead in material.
        p_self / p_pool / p_engine: probabilities that a new game is played against the current network, against a
            snapshot of the opponent pool, or against the baseline engine (depth sampled from engine_depths, where 0
            is a random player). They are normalised, can be schedules ('lin_x') and the pool is skipped while it
            is empty. The default starts with many games against the engines (strong, dense learning signal while the
            network is weak) and moves towards pure self-play.
        snapshot_every / pool_size: a snapshot of the network enters the pool every `snapshot_every` updates.
        eval_games / eval_depths: games played against each baseline at every log (the learning curve).
        """
        rng = np.random.default_rng(seed)
        if seed is not None:
            torch.manual_seed(seed)
        device, policy = self.device, self.policy
        self.pool.max_size = pool_size
        opponent_probs = [p_self, p_pool, p_engine]
        probs = np.zeros(3)

        def update_probs(progress):
            probs[:] = [schedule(p, progress) for p in opponent_probs]
            probs[:] = probs / probs.sum()
        engine_depths = np.asarray(engine_depths)

        env = ChessVecEnv(num_envs, max_plies=max_plies, open_plies=open_plies, num_threads=num_threads, seed=seed)
        T, N = num_steps, num_envs
        obs_buf = torch.empty((T, N, NUM_PLANES, 8, 8), dtype=torch.uint8, device=device)
        mask_buf = torch.empty((T, N, NUM_ACTIONS), dtype=torch.bool, device=device)
        action_buf = torch.empty((T, N), dtype=torch.long, device=device)
        log_prob_buf = torch.empty((T, N), device=device)
        value_buf = torch.empty((T, N), device=device)
        learner_buf = torch.empty((T, N), dtype=torch.bool, device=device)
        reward_buf = np.zeros((T, N), dtype=np.float32)
        mask_np_buf = np.zeros((T, N), dtype=np.float32)

        kind = np.zeros(N, dtype=np.int8)   # who plays against the learner in each env
        ref = np.zeros(N, dtype=np.int64)   # pool uid or engine depth
        color = np.zeros(N, dtype=np.int8)  # colour played by the learner (ignored in self-play)
        game_len = np.zeros(N, dtype=np.int64)

        def assign(idx):
            draw = rng.choice(3, size=len(idx), p=probs)
            if len(self.pool) == 0:
                draw[draw == POOL] = SELF
            kind[idx] = draw
            color[idx] = rng.integers(0, 2, size=len(idx))
            pool_idx = idx[draw == POOL]
            if len(pool_idx):
                ref[pool_idx] = self.pool.sample(rng, len(pool_idx))
            engine_idx = idx[draw == ENGINE]
            if len(engine_idx):
                ref[engine_idx] = rng.choice(engine_depths, size=len(engine_idx))

        window = 2000
        stats = {'len': deque(maxlen=window), 'reason': deque(maxlen=window), 'pool': deque(maxlen=window),
                 'engine': defaultdict(lambda: deque(maxlen=window))}
        updates = 0
        start_cycle, start_cycle_frames = time.time(), self.frame_idx

        try:
            env.reset()
            update_probs(min(self.frame_idx / max_frames, 1.0))
            assign(np.arange(N))
            while True:
                progress = min(self.frame_idx / max_frames, 1.0)
                update_probs(progress)
                beta = schedule(shaping, progress)
                beta_mate = schedule(mate_shaping, progress)
                beta_draw = schedule(draw_penalty, progress)
                with torch.no_grad():
                    for step in range(T):
                        obs_t = torch.from_numpy(env.obs).to(device)
                        mask_t = torch.from_numpy(env.mask).to(device)
                        action, log_prob, value = policy.act(obs_t, mask_t)
                        side = env.side.copy()
                        learner_turn = (kind == SELF) | (side == color)

                        pool_envs = np.flatnonzero((kind == POOL) & ~learner_turn)
                        for uid in np.unique(ref[pool_envs]):
                            sel = pool_envs[ref[pool_envs] == uid]
                            if uid not in self.pool.items:  # the snapshot left the pool: the learner takes over
                                kind[sel] = SELF
                                learner_turn[sel] = True
                                continue
                            sel_t = torch.from_numpy(sel).to(device)
                            action[sel_t] = self.pool.items[uid]['net'].act(obs_t[sel_t], mask_t[sel_t])[0]

                        obs_buf[step], mask_buf[step], action_buf[step] = obs_t, mask_t, action
                        log_prob_buf[step], value_buf[step] = log_prob, value
                        learner_buf[step] = torch.from_numpy(learner_turn).to(device)

                        action_np = action.cpu().numpy().astype(np.int32)
                        engine_envs = np.flatnonzero((kind == ENGINE) & ~learner_turn)
                        for depth in np.unique(ref[engine_envs]):
                            sel = engine_envs[ref[engine_envs] == depth]
                            action_np[sel] = env.engine_actions(sel, int(depth), engine_noise)[sel]

                        _, _, info = env.step(action_np)

                        reward = (info.outcome == CHECKMATE).astype(np.float32)
                        # potential-based shaping, the potential of a finished game is 0
                        phi_before = beta * info.mat_before / MATERIAL_SCALE + beta_mate * info.pot_before
                        phi_after = beta * info.mat_after / MATERIAL_SCALE + beta_mate * info.pot_after
                        phi_after = np.where(info.terminated, 0.0, phi_after)
                        reward += (gamma * phi_after - phi_before).astype(np.float32)
                        if beta_draw:
                            # bootstrapping only: finishing a game in a draw (stalemate, repetition, ...) while being
                            # ahead in material is penalised, the side that is behind is rewarded (zero-sum)
                            draw = info.terminated & (info.outcome != CHECKMATE)
                            reward -= (beta_draw * draw * np.clip(info.mat_after / 5.0, -1, 1)).astype(np.float32)
                        if len(info.final_idx):  # time limit: bootstrap with the value of the cut-off position
                            final_value = policy.get_value(torch.from_numpy(info.final_obs).to(device))
                            reward[info.final_idx] -= gamma * final_value.cpu().numpy()
                        reward_buf[step] = reward
                        mask_np_buf[step] = ~info.done
                        self.frame_idx += N
                        game_len += 1

                        done_idx = np.flatnonzero(info.done)
                        if len(done_idx):
                            for i in done_idx:
                                mate = info.outcome[i] == CHECKMATE
                                score = float(learner_turn[i]) if mate else 0.5
                                if kind[i] == SELF:
                                    stats['len'].append(game_len[i])
                                    stats['reason'].append(int(info.outcome[i]))
                                elif kind[i] == POOL:
                                    stats['pool'].append(score)
                                    self.pool.update(int(ref[i]), score)
                                else:
                                    stats['engine'][int(ref[i])].append(score)
                            game_len[done_idx] = 0
                            assign(done_idx)

                    next_value = policy.get_value(torch.from_numpy(env.obs).to(device))
                    rewards = torch.from_numpy(reward_buf).to(device)
                    masks = torch.from_numpy(mask_np_buf).to(device)
                    advantages = negamax_gae(rewards, value_buf, masks, next_value, gamma, lam)
                    returns = advantages + value_buf

                    # only the moves of the learner are trained on, the opponents' ones just shape the trajectories
                    selected = learner_buf.flatten()
                    adv = advantages.flatten()[selected]
                    adv = (adv - adv.mean()) / (adv.std(correction=0) + 1e-8)
                    batch = (obs_buf.flatten(0, 1)[selected], action_buf.flatten()[selected],
                             log_prob_buf.flatten()[selected], returns.flatten()[selected], adv,
                             mask_buf.flatten(0, 1)[selected])

                opt_info = ppo_update(batch[0], batch[1], batch[2], batch[3], batch[4], policy, self.optimizer, lr,
                                      min(batch_size, len(batch[0])), ppo_epochs, clip_range, vf_coef,
                                      schedule(ent_coef, progress), clip_grad_norm, progress, False,
                                      action_masks=batch[5], target_kl=target_kl)
                del batch
                updates += 1
                if updates % snapshot_every == 0:
                    self.pool.add(policy, self.frame_idx)

                finished = self.frame_idx >= max_frames
                if updates % log_every == 0 or finished:
                    now = time.time()
                    self.time_elapsed += now - start_cycle
                    fps = int((self.frame_idx - start_cycle_frames) / (now - start_cycle))
                    self._log(opt_info, stats, beta, fps, eval_games, eval_depths, max_plies, log_file, plot_progress)
                    start_cycle, start_cycle_frames = time.time(), self.frame_idx
                if finished:
                    break
        finally:
            env.close()

    def _log(self, opt_info, stats, beta, fps, eval_games, eval_depths, max_plies, log_file, plot_progress):
        reasons = Counter(stats['reason'])
        total = max(sum(reasons.values()), 1)
        self.history.append({
            'frames': int(self.frame_idx), 'fps': fps, 'shaping': float(beta),
            'opt': {k: float(v) for k, v in opt_info.items() if k != 'std'},
            'eval': evaluate_policy(self.policy, self.device, eval_depths, eval_games, max_plies) if eval_games else {},
            'selfplay': {
                'game_len': float(np.mean(stats['len'])) if stats['len'] else float('nan'),
                'mate': reasons[CHECKMATE] / total,
                'move_limit': reasons[MAXPLIES] / total,
                'draw': 1 - (reasons[CHECKMATE] + reasons[MAXPLIES]) / total,
                'games': len(stats['len']),
            },
            'vs_pool': float(np.mean(stats['pool'])) if stats['pool'] else float('nan'),
            'vs_engine': {int(d): float(np.mean(s)) for d, s in sorted(stats['engine'].items()) if len(s)},
            'pool_size': len(self.pool),
        })
        if log_file is not None:
            self.save(log_file)
        if plot_progress:
            plot_history(self.history, self.time_elapsed)
        else:
            print_summary(self.history, self.time_elapsed)


def plot_history(history, time_elapsed=None):
    """Learning curves of a SelfPlayPPO run (SelfPlayPPO.history)."""
    frames = [h['frames'] for h in history]
    fig, axes = plt.subplots(2, 3, figsize=(16, 7), constrained_layout=True)

    ax = axes[0, 0]
    for name in history[-1]['eval']:
        ax.plot(frames, [h['eval'].get(name, np.nan) for h in history], label='vs ' + name)
    ax.set_title('score against the baselines (1 = wins all)')
    ax.set_ylim(-0.02, 1.02)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(frames, [h['selfplay']['mate'] for h in history], label='checkmate')
    ax.plot(frames, [h['selfplay']['move_limit'] for h in history], label='move limit')
    ax.plot(frames, [h['selfplay']['draw'] for h in history], label='other draws')
    ax.set_title('how the self-play games end')
    ax.legend()

    axes[0, 2].plot(frames, [h['selfplay']['game_len'] for h in history])
    axes[0, 2].set_title('self-play game length (half-moves)')

    ax = axes[1, 0]
    ax.plot(frames, [h['opt']['critic_loss'] * h['opt']['vf_coef'] for h in history], label='critic')
    ax.plot(frames, [h['opt']['actor_loss'] for h in history], label='actor')
    ax.set_title('losses')
    ax.legend()

    axes[1, 1].plot(frames, [h['opt'].get('entropy', np.nan) for h in history])
    axes[1, 1].set_title('policy entropy')

    ax = axes[1, 2]
    ax.plot(frames, [h['opt']['approx_kl'] for h in history], label='approx kl')
    ax.plot(frames, [h['opt']['clip_fraction'] for h in history], label='clip fraction')
    ax.plot(frames, [h['shaping'] for h in history], label='shaping weight')
    ax.set_title('update size')
    ax.legend()

    clear_output(True)
    display(fig)
    plt.close(fig)
    print_summary(history, time_elapsed)


def print_summary(history, time_elapsed=None):
    last = history[-1]
    print('frames: %i   fps: %i   logs: %i%s' % (last['frames'], last['fps'], len(history),
                                                  '   time: %i min' % (time_elapsed / 60) if time_elapsed else ''))
    print('eval  :', ', '.join('%s %.2f' % kv for kv in last['eval'].items()))
    print('train :', ', '.join('depth%i %.2f' % kv for kv in last['vs_engine'].items()),
          '| pool %.2f (size %i)' % (last['vs_pool'], last['pool_size']))
    selfplay = last['selfplay']
    print('self-play: length %.0f, checkmate %.2f, move limit %.2f, other draws %.2f | entropy %.2f, kl %.4f' % (
        selfplay['game_len'], selfplay['mate'], selfplay['move_limit'], selfplay['draw'],
        last['opt'].get('entropy', float('nan')), last['opt']['approx_kl']))
