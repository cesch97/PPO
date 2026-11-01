"""
Vectorized chess environment.

The rules, the move generation and the observation encoding live in `chess_engine.c`, which is compiled the first time
this module is imported (the result is cached in `PPO/_build`). All the boards of the batch live in a single numpy
buffer and are stepped by plain C code from a pool of threads (ctypes releases the GIL): there is a single Python
interpreter and no copy of the state, so the memory footprint is a few KB per environment.

Everything is expressed from the point of view of the player to move (the board is mirrored when black moves):
 - observation: uint8 array (NUM_PLANES, 8, 8) with values in {0, 255} (scalar planes use the whole range),
   so a single network can play both colours;
 - action: integer in [0, 4672) = from_square * 73 + move_type (AlphaZero encoding: 56 sliding moves, 8 knight moves,
   9 under-promotions; promotions to queen are encoded as the corresponding pawn move);
 - rewards are given to the player who has just moved, so a game is zero-sum and negamax values apply.
"""
import ctypes
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np

NUM_PLANES = 21
NUM_ACTIONS = 4672
MAX_PLIES_LIMIT = 511
PLANE_NAMES = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k', 'en passant', 'castle K', 'castle Q',
               'opp castle K', 'opp castle Q', 'half-move clock', 'repeated once', 'repeated twice', 'game progress']

ONGOING, CHECKMATE, STALEMATE, INSUFFICIENT, FIFTY, REPETITION, MAXPLIES = range(7)
OUTCOME_NAMES = ['ongoing', 'checkmate', 'stalemate', 'insufficient material', 'fifty-move rule',
                 'threefold repetition', 'move limit']

_SOURCE = Path(__file__).with_name('chess_engine.c')
_BUILD_DIR = Path(__file__).with_name('_build')


def _compilers():
    if os.environ.get('CC'):
        yield os.environ['CC'].split()
    for name in ('cc', 'gcc', 'clang'):
        if shutil.which(name):
            yield [name]
    # the `ziglang` wheel ships a C compiler, so no system toolchain is needed
    yield [sys.executable, '-m', 'ziglang', 'cc']


def _build_library():
    source = _SOURCE.read_bytes()
    tag = hashlib.sha1(source + platform.machine().encode() + sys.platform.encode()).hexdigest()[:12]
    target = _BUILD_DIR / ('chess_engine_%s.so' % tag)
    if target.exists():
        return target
    _BUILD_DIR.mkdir(exist_ok=True)
    errors = []
    for compiler in _compilers():
        with tempfile.NamedTemporaryFile(dir=_BUILD_DIR, suffix='.so', delete=False) as tmp:
            tmp_path = Path(tmp.name)
        for arch in (['-march=native'], []):
            cmd = compiler + ['-O3', *arch, '-std=c11', '-shared', '-fPIC', str(_SOURCE), '-o', str(tmp_path)]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
            except OSError as e:
                errors.append('%s: %s' % (' '.join(compiler), e))
                break
            if result.returncode == 0:
                os.replace(tmp_path, target)  # atomic, safe if several processes build at the same time
                return target
            errors.append('%s: %s' % (' '.join(cmd), result.stderr.strip()))
        tmp_path.unlink(missing_ok=True)
    raise RuntimeError('could not compile the chess engine:\n' + '\n'.join(errors))


def _load_library():
    lib = ctypes.CDLL(str(_build_library()))
    p, i, u64 = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64
    signatures = {
        'ce_init': (None, []),
        'ce_board_size': (i, []),
        'ce_num_planes': (i, []),
        'ce_num_actions': (i, []),
        'ce_max_hist': (i, []),
        'ce_encode_move': (i, [i, i, i, i]),
        'ce_seed': (None, [p, i, u64]),
        'ce_reset': (None, [p, p, i, i, i, i]),
        'ce_step': (None, [p, p, i, i, p, i, p, p, p, p, p]),
        'ce_observe': (None, [p, p, i, i, i, p, p, p]),
        'ce_engine_action': (None, [p, p, i, i, i, i, p]),
        'ce_set_fen': (i, [p, i, ctypes.c_char_p]),
        'ce_get_fen': (None, [p, i, p]),
        'ce_legal_moves': (i, [p, i, p]),
        'ce_apply_move': (i, [p, i, ctypes.c_uint32, i]),
        'ce_perft': (u64, [p, i, i]),
        'ce_in_check': (i, [p, i]),
        'ce_status': (i, [p, i, i]),
    }
    for name, (restype, argtypes) in signatures.items():
        fn = getattr(lib, name)
        fn.restype, fn.argtypes = restype, argtypes
    lib.ce_init()
    assert lib.ce_num_planes() == NUM_PLANES and lib.ce_num_actions() == NUM_ACTIONS
    return lib


_LIB = _load_library()
_BOARD_SIZE = _LIB.ce_board_size()


def _ptr(array):
    return None if array is None else array.ctypes.data


def encode_move(side, from_sq, to_sq, promotion=0):
    """Action index of a move; `side` is the colour moving (0 white, 1 black), `promotion` 0 or 1..4 = N, B, R, Q."""
    return _LIB.ce_encode_move(side, from_sq, to_sq, promotion)


class StepInfo(NamedTuple):
    outcome: np.ndarray       # (N,) int8, game status after the move (see OUTCOME_NAMES)
    mat_before: np.ndarray    # (N,) int16, material balance (pawns) of the player who moved, before the move
    mat_after: np.ndarray     # (N,) int16, same after the move
    pot_before: np.ndarray    # (N,) float32, mating-technique potential in [-1, 1] (see chess_engine.c), same view
    pot_after: np.ndarray     # (N,) float32, same after the move
    done: np.ndarray          # (N,) bool, the game is over (the env has been reset if reset=True)
    terminated: np.ndarray    # (N,) bool, a real end of the game (mate or draw)
    truncated: np.ndarray     # (N,) bool, the move limit was reached
    final_idx: np.ndarray     # indices of the truncated envs
    final_obs: Optional[np.ndarray]  # observations of the truncated envs before they were reset


class ChessVecEnv:
    """
    A batch of `num_envs` chess games.

    max_plies: games are truncated after this number of half-moves (counted from the start of the game).
    open_plies: (min, max) number of random half-moves played at every reset, to diversify the starting positions.
    num_threads: threads used to step the batch (None = up to 16, scaled to the batch size).
    """

    def __init__(self, num_envs, max_plies=200, open_plies=(0, 8), num_threads=None, seed=None):
        if not 0 < max_plies <= MAX_PLIES_LIMIT:
            raise ValueError('max_plies must be in [1, %i]' % MAX_PLIES_LIMIT)
        self.num_envs = num_envs
        self.max_plies = max_plies
        self.open_plies = tuple(open_plies)
        if num_threads is None:
            num_threads = min(16, os.cpu_count() or 1, max(1, num_envs // 16))
        self.num_threads = max(1, num_threads)
        self._pool = ThreadPoolExecutor(self.num_threads) if self.num_threads > 1 else None

        self._boards = np.zeros(num_envs * _BOARD_SIZE, dtype=np.uint8)
        seed = int(np.random.SeedSequence(seed).generate_state(1, dtype=np.uint64)[0])
        for i in range(num_envs):
            _LIB.ce_seed(_ptr(self._boards), i, (seed + i * 0x9E3779B97F4A7C15) & (2 ** 64 - 1))

        self.obs = np.zeros((num_envs, NUM_PLANES, 8, 8), dtype=np.uint8)
        self.mask = np.zeros((num_envs, NUM_ACTIONS), dtype=np.bool_)
        self.side = np.zeros(num_envs, dtype=np.int8)  # colour to move: 0 white, 1 black
        self._outcome = np.zeros(num_envs, dtype=np.int8)
        self._mat_before = np.zeros(num_envs, dtype=np.int16)
        self._mat_after = np.zeros(num_envs, dtype=np.int16)
        self._pot_before = np.zeros(num_envs, dtype=np.float32)
        self._pot_after = np.zeros(num_envs, dtype=np.float32)
        self._actions = np.zeros(num_envs, dtype=np.int32)

    # ----------------------------------------------------------------------------------------------- threading
    def _run(self, fn, idx=None, min_chunk=16):
        """Calls fn(idx_ptr, start, end) on chunks of the batch (or of `idx`) from the thread pool."""
        n = self.num_envs if idx is None else len(idx)
        if n == 0:
            return
        idx_ptr = _ptr(idx)
        chunks = min(self.num_threads, max(1, n // min_chunk))
        if self._pool is None or chunks <= 1:
            fn(idx_ptr, 0, n)
            return
        bounds = np.linspace(0, n, chunks + 1).astype(int)
        futures = [self._pool.submit(fn, idx_ptr, int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:])]
        for future in futures:
            future.result()

    def _observe(self, idx=None):
        self._run(lambda ip, a, b: _LIB.ce_observe(self._board_ptr(), ip, a, b, self.max_plies, _ptr(self.obs),
                                                    _ptr(self.mask), _ptr(self.side)), idx)

    def close(self):
        if getattr(self, '_pool', None) is not None:
            self._pool.shutdown()
            self._pool = None

    def __del__(self):
        self.close()

    # ----------------------------------------------------------------------------------------------- gym-like API
    def reset(self, idx=None):
        """Resets all the envs (or the ones in `idx`) and returns (obs, mask) of the whole batch."""
        if idx is not None:
            idx = np.ascontiguousarray(idx, dtype=np.int32)
        lo, hi = self.open_plies
        self._run(lambda ip, a, b: _LIB.ce_reset(self._board_ptr(), ip, a, b, lo, hi), idx)
        self._observe(idx)
        return self.obs, self.mask

    def step(self, actions, reset=True):
        """
        Plays one move in every env. `actions` are indices of legal moves (see `mask`).
        Returns (obs, mask, StepInfo): obs and mask describe the new positions (the envs that finished have already
        been reset if reset=True) and are views of buffers that are overwritten by the next call.
        """
        actions = np.ascontiguousarray(actions, dtype=np.int32)
        self._run(lambda ip, a, b: _LIB.ce_step(self._board_ptr(), ip, a, b, _ptr(actions), self.max_plies,
                                                _ptr(self._outcome), _ptr(self._mat_before), _ptr(self._mat_after),
                                                _ptr(self._pot_before), _ptr(self._pot_after)))
        illegal = np.flatnonzero(self._outcome < 0)
        if len(illegal):
            raise ValueError('illegal actions in envs %s' % illegal[:10].tolist())
        self._observe()
        outcome = self._outcome.copy()
        done = outcome != ONGOING
        truncated = outcome == MAXPLIES
        final_idx = np.flatnonzero(truncated)
        final_obs = self.obs[final_idx].copy() if len(final_idx) else None
        info = StepInfo(outcome, self._mat_before.copy(), self._mat_after.copy(), self._pot_before.copy(),
                        self._pot_after.copy(), done, done & ~truncated, truncated, final_idx, final_obs)
        if reset and done.any():
            self.reset(np.flatnonzero(done))
        return self.obs, self.mask, info

    def engine_actions(self, idx=None, depth=0, noise=0):
        """
        Actions of the built-in baseline player for the envs in `idx` (all by default), returned in a (N,) array whose
        other entries are undefined. depth 0 plays uniformly random legal moves; depth d >= 1 is an alpha-beta search
        (material + light positional evaluation, quiescence search) with `noise` centipawns of random tie-breaking.
        """
        if idx is not None:
            idx = np.ascontiguousarray(idx, dtype=np.int32)
        self._run(lambda ip, a, b: _LIB.ce_engine_action(self._board_ptr(), ip, a, b, depth, noise,
                                                         _ptr(self._actions)), idx, min_chunk=1 if depth else 16)
        return self._actions

    # ----------------------------------------------------------------------------------------------- inspection
    def _board_ptr(self):
        return _ptr(self._boards)

    def set_fen(self, i, fen):
        if _LIB.ce_set_fen(self._board_ptr(), i, fen.encode()) != 0:
            raise ValueError('invalid FEN: %s' % fen)
        self._observe(np.array([i], dtype=np.int32))

    def fen(self, i):
        buf = ctypes.create_string_buffer(128)
        _LIB.ce_get_fen(self._board_ptr(), i, ctypes.addressof(buf))
        return buf.value.decode()

    def legal_moves(self, i):
        """Legal moves of env i as a list of (from_sq, to_sq, promotion) with promotion 0 or 1..4 = N, B, R, Q."""
        buf = (ctypes.c_uint32 * 256)()
        n = _LIB.ce_legal_moves(self._board_ptr(), i, ctypes.addressof(buf))
        return [(m & 63, (m >> 6) & 63, (m >> 12) & 7) for m in buf[:n]]

    def play(self, i, from_sq, to_sq, promotion=0):
        """Plays a legal move in env i (no reset); returns the game status, -1 if the move is illegal."""
        move = from_sq | to_sq << 6 | promotion << 12
        status = _LIB.ce_apply_move(self._board_ptr(), i, move, self.max_plies)
        if status >= 0:
            self._observe(np.array([i], dtype=np.int32))
        return status

    def status(self, i):
        return _LIB.ce_status(self._board_ptr(), i, self.max_plies)

    def in_check(self, i):
        return bool(_LIB.ce_in_check(self._board_ptr(), i))

    def perft(self, i, depth):
        return _LIB.ce_perft(self._board_ptr(), i, depth)
