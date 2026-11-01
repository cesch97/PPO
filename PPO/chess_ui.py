"""
Notebook helpers to watch two agents play and to play against an agent.

`GameSession` is the game logic (rules are the ones of the vectorized engine, python-chess only parses the moves typed
by the user and draws the board); `watch_game` and `HumanGame` are the front-ends.
"""
import time

import chess
import chess.pgn
import chess.svg
import numpy as np
from IPython.display import HTML, display

from PPO.chess_env import CHECKMATE, MAX_PLIES_LIMIT, ONGOING, OUTCOME_NAMES, ChessVecEnv
from PPO.selfplay import action_to_move

PROMOTION_PIECE = {1: chess.KNIGHT, 2: chess.BISHOP, 3: chess.ROOK, 4: chess.QUEEN}
PIECE_PROMOTION = {v: k for k, v in PROMOTION_PIECE.items()}


class GameSession:
    """One game between any combination of humans and players (objects with act(env, idx), see PPO.selfplay)."""

    def __init__(self):
        self.env = ChessVecEnv(1, max_plies=MAX_PLIES_LIMIT, open_plies=(0, 0), num_threads=1)
        self.new_game()

    def new_game(self, fen=chess.STARTING_FEN):
        self.env.set_fen(0, fen)
        self.board = chess.Board(fen)
        self.start_fen = fen
        self.moves = []
        self.status = ONGOING

    @property
    def over(self):
        return self.status != ONGOING

    @property
    def turn(self):
        return self.board.turn

    def push(self, move):
        """Plays a python-chess Move (it must be legal)."""
        if self.over:
            raise ValueError('the game is over')
        status = self.env.play(0, move.from_square, move.to_square, PIECE_PROMOTION.get(move.promotion, 0))
        if status < 0:
            raise ValueError('illegal move %s' % move.uci())
        self.board.push(move)
        self.moves.append(move)
        self.status = status

    def push_text(self, text):
        """Plays a move written in SAN (Nf3, exd5, O-O, e8=Q) or UCI (g1f3, e7e8q)."""
        text = text.strip()
        try:
            move = self.board.parse_san(text)
        except ValueError:
            try:
                move = self.board.parse_uci(text.lower())
            except ValueError:
                raise ValueError("'%s' is not a legal move here" % text) from None
        self.push(move)
        return move

    def agent_move(self, player):
        """Asks `player` for a move in the current position and plays it; returns the move in SAN."""
        action = int(player.act(self.env, np.array([0], dtype=np.int32))[0])
        from_sq, to_sq, promotion = action_to_move(self.env, 0, action)
        move = chess.Move(from_sq, to_sq, PROMOTION_PIECE.get(promotion))
        san = self.board.san(move)
        self.push(move)
        return san

    def undo(self, plies=2):
        moves = self.moves[:max(len(self.moves) - plies, 0)]
        self.new_game(self.start_fen)
        for move in moves:
            self.push(move)

    def result_text(self):
        if self.status == ONGOING:
            return 'White to move' if self.board.turn == chess.WHITE else 'Black to move'
        if self.status == CHECKMATE:
            return 'Checkmate: %s wins' % ('Black' if self.board.turn == chess.WHITE else 'White')
        return 'Draw (%s)' % OUTCOME_NAMES[self.status]

    def pgn(self, white='White', black='Black'):
        game = chess.pgn.Game()
        if self.start_fen != chess.STARTING_FEN:
            game.setup(self.start_fen)
        game.add_line(self.moves)
        game.headers['White'], game.headers['Black'] = white, black
        if self.status == CHECKMATE:
            game.headers['Result'] = '0-1' if self.board.turn == chess.WHITE else '1-0'
        elif self.over:
            game.headers['Result'] = '1/2-1/2'
        return str(game)

    def svg(self, size=440, flipped=False):
        last = self.moves[-1] if self.moves else None
        check = self.board.king(self.board.turn) if self.board.is_check() else None
        return chess.svg.board(self.board, lastmove=last, check=check, flipped=flipped, size=size)


def watch_game(white, black, delay=0.4, size=440, names=('White', 'Black'), max_plies=300):
    """Plays one game between two players showing the board in the notebook; returns the GameSession (see .pgn())."""
    session = GameSession()
    display(HTML('<b>%s</b> (white) vs <b>%s</b> (black)' % names))
    handle = display(HTML(session.svg(size)), display_id=True)
    caption = display(HTML(session.result_text()), display_id=True)
    while not session.over and len(session.moves) < max_plies:
        player = white if session.turn == chess.WHITE else black
        san = session.agent_move(player)
        if handle is not None:  # None outside of a notebook
            handle.update(HTML(session.svg(size)))
            caption.update(HTML('%i. %s &nbsp; %s' % ((len(session.moves) + 1) // 2, san, session.result_text())))
        time.sleep(delay)
    return session


class HumanGame:
    """
    Play against an agent inside the notebook: type your move (SAN like Nf3, O-O, exd5, e8=Q or UCI like g1f3) in the
    box and press Enter, or pick it from the list of legal moves.
    """

    def __init__(self, agent_player, human_color='white', size=480):
        import ipywidgets as widgets
        self.agent = agent_player
        self.size = size
        self.session = GameSession()
        self.message = ''
        self.human_color = chess.WHITE

        self.board_view = widgets.HTML()
        self.status = widgets.HTML()
        self.moves_view = widgets.HTML()
        self.color = widgets.Dropdown(options=['white', 'black', 'random'], value=human_color, description='You play',
                                      layout=widgets.Layout(width='190px'))
        self.new_button = widgets.Button(description='New game', button_style='success',
                                         layout=widgets.Layout(width='110px'))
        self.undo_button = widgets.Button(description='Undo', layout=widgets.Layout(width='80px'))
        self.entry = widgets.Text(placeholder='your move, e.g. e4, Nf3, O-O, e7e8q', continuous_update=False,
                                  layout=widgets.Layout(width='260px'))
        self.legal = widgets.Dropdown(options=[''], description='Legal moves', layout=widgets.Layout(width='260px'))
        self.new_button.on_click(lambda _: self.new_game())
        self.undo_button.on_click(lambda _: self.undo())
        self.entry.observe(self._on_text, names='value')
        self.legal.observe(self._on_pick, names='value')
        self.ui = widgets.VBox([
            widgets.HBox([self.color, self.new_button, self.undo_button]),
            self.board_view, self.status,
            widgets.HBox([self.entry, self.legal]),
            self.moves_view])
        self.new_game()

    def _show(self):
        session = self.session
        self.board_view.value = session.svg(self.size, flipped=self.human_color == chess.BLACK)
        text = session.result_text()
        if not session.over and session.turn == self.human_color:
            text += ' (your turn)'
        self.status.value = '<b>%s</b>%s' % (text, '' if not self.message else ' &nbsp; <i>%s</i>' % self.message)
        board = chess.Board(session.start_fen)
        san = []
        for i, move in enumerate(session.moves):
            san.append(('%i. ' % (i // 2 + 1) if i % 2 == 0 else '') + board.san(move))
            board.push(move)
        self.moves_view.value = '<small>%s</small>' % ' '.join(san)
        self._set_legal()

    def _set_legal(self):
        session = self.session
        self.legal.unobserve(self._on_pick, names='value')
        if session.over or session.turn != self.human_color:
            options = ['']
        else:
            options = [''] + sorted(session.board.san(m) for m in session.board.legal_moves)
        self.legal.options = options
        self.legal.value = ''
        self.legal.observe(self._on_pick, names='value')

    def _agent_turn(self):
        if not self.session.over and self.session.turn != self.human_color:
            self.message = 'agent plays %s' % self.session.agent_move(self.agent)

    def new_game(self):
        color = self.color.value
        if color == 'random':
            color = ['white', 'black'][np.random.randint(2)]
        self.human_color = chess.WHITE if color == 'white' else chess.BLACK
        self.session.new_game()
        self.message = ''
        self._agent_turn()
        self._show()

    def undo(self):
        """Takes back the last human move together with the agent's reply."""
        session = self.session
        n = len(session.moves)
        keep = n - 2 if session.turn == self.human_color else n - 1
        if session.over and session.turn != self.human_color:
            keep = n - 1
        # never remove the agent's opening move when playing black
        first_human = 0 if self.human_color == chess.WHITE else 1
        if keep < first_human:
            return
        session.undo(n - keep)
        self.message = 'move taken back'
        self._show()

    def _play(self, text):
        session = self.session
        if session.over or session.turn != self.human_color:
            return
        try:
            session.push_text(text)
        except ValueError as e:
            self.message = str(e)
            self._show()
            return
        self.message = ''
        self._agent_turn()
        self._show()

    def _on_text(self, change):
        if change['new']:
            self.entry.unobserve(self._on_text, names='value')
            self.entry.value = ''
            self.entry.observe(self._on_text, names='value')
            self._play(change['new'])

    def _on_pick(self, change):
        if change['new']:
            self._play(change['new'])

    def _ipython_display_(self):
        display(self.ui)
