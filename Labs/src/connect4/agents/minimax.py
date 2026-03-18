import random

from .base import BaseAgent


class MinimaxAgent(BaseAgent):
    """Minimax agent without alpha-beta pruning."""

    def __init__(self, ai_player=2, depth=4):
        super().__init__(ai_player=ai_player)
        self.depth = depth

    def get_best_move(self, game):
        self.nodes_visited = 0
        best_score = -float("inf")
        best_col = random.choice(game.actions())

        for col in game.actions():
            row = game.drop_piece(col)
            score = self._minimax(game, self.depth - 1, False)
            game.undo_move(col, row)
            if score > best_score:
                best_score = score
                best_col = col

        return best_col, self.nodes_visited

    def select_action(self, game):
        best_col, _ = self.get_best_move(game)
        return best_col

    def _minimax(self, game, depth, is_maximizing):
        self.nodes_visited += 1

        if game.is_terminal():
            return game.get_terminal_score(self.ai_player)

        if depth == 0:
            return 0

        if is_maximizing:
            max_eval = -float("inf")
            for col in game.actions():
                row = game.drop_piece(col)
                eval_score = self._minimax(game, depth - 1, False)
                game.undo_move(col, row)
                max_eval = max(max_eval, eval_score)
            return max_eval

        min_eval = float("inf")
        for col in game.actions():
            row = game.drop_piece(col)
            eval_score = self._minimax(game, depth - 1, True)
            game.undo_move(col, row)
            min_eval = min(min_eval, eval_score)
        return min_eval
