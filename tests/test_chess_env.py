import chess
import numpy as np
import pytest

from PPO.chess_env import CHECKMATE, ChessVecEnv, NUM_ACTIONS, NUM_PLANES, ONGOING, STALEMATE, encode_move

PERFT = [
    ('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1', [20, 400, 8902, 197281, 4865609]),
    ('r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1', [48, 2039, 97862, 4085603]),
    ('8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1', [14, 191, 2812, 43238, 674624]),
    ('r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1', [6, 264, 9467, 422333]),
    ('rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8', [44, 1486, 62379, 2103487]),
]


def to_action(board, move):
    return encode_move(0 if board.turn == chess.WHITE else 1, move.from_square, move.to_square,
                       move.promotion - 1 if move.promotion else 0)


@pytest.mark.parametrize('fen,expected', PERFT)
def test_perft(fen, expected):
    env = ChessVecEnv(1, num_threads=1)
    env.set_fen(0, fen)
    assert [env.perft(0, d + 1) for d in range(len(expected))] == expected


def test_random_games_match_python_chess():
    """Positions, action masks and game results agree with python-chess along random games."""
    n = 16
    env = ChessVecEnv(n, max_plies=400, open_plies=(0, 0), seed=1, num_threads=4)
    env.reset()
    rng = np.random.default_rng(0)
    boards = [chess.Board() for _ in range(n)]
    finished = 0
    for _ in range(300):
        actions = np.zeros(n, dtype=np.int32)
        chosen = []
        for i, board in enumerate(boards):
            legal = list(board.legal_moves)
            assert env.fen(i).split(' ')[:4] == board.fen(en_passant='xfen').split(' ')[:4]
            expected = {to_action(board, m) for m in legal}
            assert set(np.flatnonzero(env.mask[i])) == expected and len(expected) == len(legal)
            assert env.side[i] == (0 if board.turn == chess.WHITE else 1)
            move = legal[rng.integers(len(legal))]
            chosen.append(move)
            actions[i] = to_action(board, move)
        _, _, info = env.step(actions)
        for i, board in enumerate(boards):
            board.push(chosen[i])
            if board.is_checkmate():
                assert info.outcome[i] == CHECKMATE
            elif board.is_stalemate():
                assert info.outcome[i] == STALEMATE
            over = (board.is_game_over(claim_draw=False) or board.is_repetition(3)
                    or board.halfmove_clock >= 100)
            assert bool(info.done[i]) == over
            if over:
                finished += 1
                boards[i] = chess.Board()
            else:
                assert info.outcome[i] == ONGOING
    assert finished > 0


def test_action_encoding_is_injective_on_legal_moves():
    env = ChessVecEnv(1, num_threads=1)
    fens = [p[0] for p in PERFT] + ['4k3/P6P/8/8/8/8/p6p/4K3 w - - 0 1', '4k3/P6P/8/8/8/8/p6p/4K3 b - - 0 1']
    for fen in fens:
        env.set_fen(0, fen)
        actions = [encode_move(env.side[0], *m) for m in env.legal_moves(0)]
        assert len(set(actions)) == len(actions) and all(0 <= a < NUM_ACTIONS for a in actions)
        assert set(actions) == set(np.flatnonzero(env.mask[0]))


def test_observation_is_from_the_point_of_view_of_the_mover():
    env = ChessVecEnv(2, num_threads=1)
    env.set_fen(0, 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
    env.set_fen(1, 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1')
    assert env.obs.shape == (2, NUM_PLANES, 8, 8)
    assert np.array_equal(env.obs[0, :20], env.obs[1, :20])
    assert np.array_equal(env.mask[0], env.mask[1])
    assert env.obs[0, 0, 1].tolist() == [255] * 8  # own pawns on the second rank
    assert env.obs[0, 6, 6].tolist() == [255] * 8  # opponent pawns on the seventh rank


def test_step_semantics_mate_truncation_and_reset():
    env = ChessVecEnv(2, max_plies=5, open_plies=(0, 0), num_threads=1)
    env.set_fen(0, '6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1')  # Ra8#
    env.set_fen(1, 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
    _, _, info = env.step(np.array([encode_move(0, 0, 56), encode_move(0, 12, 28)]))
    assert info.outcome[0] == CHECKMATE and info.done[0] and info.terminated[0]
    assert not info.done[1] and info.mat_before[1] == info.mat_after[1]
    assert env.fen(0).startswith('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w')  # auto reset

    with pytest.raises(ValueError):
        env.step(np.array([0, 0]))

    env.reset()
    for _ in range(5):
        _, _, info = env.step(env.engine_actions(depth=0))
    assert info.truncated.all() and info.final_obs.shape == (2, NUM_PLANES, 8, 8)
    assert np.array_equal(info.final_idx, [0, 1])


def test_engine_finds_mate_and_captures_hanging_queen():
    env = ChessVecEnv(3, num_threads=1)
    env.set_fen(0, '6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1')
    env.set_fen(1, 'rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2')  # fool's mate: ...Qh4#
    env.set_fen(2, '4k3/8/8/3q4/8/8/8/3QK3 w - - 0 1')
    for depth in (1, 2, 3):
        actions = env.engine_actions(depth=depth)
        assert actions[0] == encode_move(0, 0, 56)
        assert actions[1] == encode_move(1, 59, 31)
        assert actions[2] == encode_move(0, 3, 35)


def test_potentials_are_consistent_between_consecutive_moves():
    """The potential after a move (mover's view) is minus the potential before the reply (opponent's view)."""
    env = ChessVecEnv(16, max_plies=400, open_plies=(0, 10), seed=5, num_threads=2)
    env.reset()
    previous = None
    for _ in range(80):
        _, _, info = env.step(env.engine_actions(depth=0))
        if previous is not None:
            keep = ~previous.done
            assert np.allclose(info.pot_before[keep], -previous.pot_after[keep])
            assert np.array_equal(info.mat_before[keep], -previous.mat_after[keep])
        previous = info
    assert np.abs(info.pot_after).max() <= 1.0


def test_mate_potential_rewards_cornering_the_king():
    env = ChessVecEnv(2, num_threads=1)
    env.set_fen(0, '8/8/8/3k4/8/8/8/R3K3 w - - 0 1')  # king in the centre
    env.set_fen(1, '7k/8/8/8/8/8/8/R3K3 w - - 0 1')   # king in the corner
    _, _, info = env.step(np.array([encode_move(0, 0, 8)] * 2), reset=False)
    assert info.pot_after[1] > info.pot_after[0] > 0
