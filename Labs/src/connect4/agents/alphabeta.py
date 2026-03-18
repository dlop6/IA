import random

from ..game import Connect4
from ..heuristics import evaluate
from .base import BaseAgent


class AlphaBetaAgent(BaseAgent):
    """Minimax agent with alpha-beta pruning and no heuristic leaves."""

    def __init__(self, ai_player=Connect4.PLAYER2, depth=4):
        super().__init__(ai_player=ai_player)
        self.depth = depth

    def get_best_move(self, game):
        self.nodes_visited = 0
        best_score = -float("inf")
        best_col = random.choice(game.actions())
        alpha = -float("inf")
        beta = float("inf")

        for col in game.actions():
            row = game.drop_piece(col)
            score = self._alphabeta(game, self.depth - 1, alpha, beta, False)
            game.undo_move(col, row)
            if score > best_score:
                best_score = score
                best_col = col
            alpha = max(alpha, best_score)

        return best_col, self.nodes_visited

    def select_action(self, game):
        best_col, _ = self.get_best_move(game)
        return best_col

    def _alphabeta(self, game, depth, alpha, beta, is_maximizing):
        self.nodes_visited += 1

        if game.is_terminal():
            return game.get_terminal_score(self.ai_player)

        if depth == 0:
            return 0

        if is_maximizing:
            max_eval = -float("inf")
            for col in game.actions():
                row = game.drop_piece(col)
                eval_score = self._alphabeta(game, depth - 1, alpha, beta, False)
                game.undo_move(col, row)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval

        min_eval = float("inf")
        for col in game.actions():
            row = game.drop_piece(col)
            eval_score = self._alphabeta(game, depth - 1, alpha, beta, True)
            game.undo_move(col, row)
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval


class SmartAlphaBetaAgent(BaseAgent):
    """Alpha-beta agent with the notebook's heuristic leaf evaluation."""

    def __init__(self, ai_player=Connect4.PLAYER2, depth=6):
        super().__init__(ai_player=ai_player)
        self.depth = depth

    def get_best_move(self, game):
        self.nodes_visited = 0
        best_score = -float("inf")
        best_col = random.choice(game.actions())
        alpha = -float("inf")
        beta = float("inf")
        actions = self._ordered_actions(game)

        for col in actions:
            row = game.drop_piece(col)
            score = self._alphabeta(game, self.depth - 1, alpha, beta, False)
            game.undo_move(col, row)
            if score > best_score:
                best_score = score
                best_col = col
            alpha = max(alpha, best_score)

        return best_col, self.nodes_visited

    def select_action(self, game):
        best_col, _ = self.get_best_move(game)
        return best_col

    def _ordered_actions(self, game):
        actions = game.actions()
        center = Connect4.COLS // 2
        actions.sort(key=lambda col: abs(col - center))
        return actions

    def _alphabeta(self, game, depth, alpha, beta, is_maximizing):
        self.nodes_visited += 1

        if game.is_terminal():
            return game.get_terminal_score(self.ai_player)

        if depth == 0:
            return evaluate(game, self.ai_player)

        actions = self._ordered_actions(game)

        if is_maximizing:
            max_eval = -float("inf")
            for col in actions:
                row = game.drop_piece(col)
                eval_score = self._alphabeta(game, depth - 1, alpha, beta, False)
                game.undo_move(col, row)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval

        min_eval = float("inf")
        for col in actions:
            row = game.drop_piece(col)
            eval_score = self._alphabeta(game, depth - 1, alpha, beta, True)
            game.undo_move(col, row)
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval
