import chess

from PPO.chess_ui import GameSession, HumanGame, watch_game
from PPO.selfplay import EnginePlayer


def test_session_plays_san_and_uci_and_detects_mate():
    s = GameSession()
    for move in ['f3', 'e5', 'g2g4', 'Qh4#']:
        s.push_text(move)
    assert s.over and s.result_text() == 'Checkmate: Black wins'
    assert '0-1' in s.pgn('a', 'b')
    s.undo(1)
    assert not s.over and len(s.moves) == 3


def test_session_rejects_illegal_moves():
    s = GameSession()
    for bad in ['e5', 'Ke2', 'zzz']:
        try:
            s.push_text(bad)
        except ValueError:
            continue
        raise AssertionError(bad)
    assert not s.moves


def test_promotion_and_castling_through_the_engine():
    s = GameSession()
    s.new_game('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1')
    s.push_text('O-O')
    s.push_text('O-O-O')
    s.new_game('4k3/P7/8/8/8/8/8/4K3 w - - 0 1')
    s.push_text('a8=N')
    assert s.board.piece_at(chess.A8).piece_type == chess.KNIGHT
    assert s.env.fen(0).startswith('N3k3/')


def test_watch_game_and_human_game():
    session = watch_game(EnginePlayer(1), EnginePlayer(0), delay=0, max_plies=30)
    assert session.over or len(session.moves) == 30

    game = HumanGame(EnginePlayer(0), human_color='black')
    assert len(game.session.moves) == 1  # the agent opened
    game._play('e5')
    assert len(game.session.moves) == 3
    game.undo()
    assert len(game.session.moves) == 1
    game._play('nonsense')
    assert 'not a legal move' in game.message
    game.color.value = 'white'
    game.new_game()
    assert game.session.moves == [] and game.human_color == chess.WHITE
