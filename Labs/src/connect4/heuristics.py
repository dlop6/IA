import numpy as np

from .game import Connect4


def evaluate_window(window, ai_player):
    """Score a 4-cell window from the perspective of ai_player."""
    opponent = Connect4.PLAYER1 if ai_player == Connect4.PLAYER2 else Connect4.PLAYER2
    score = 0

    ai_count = np.count_nonzero(window == ai_player)
    opp_count = np.count_nonzero(window == opponent)
    empty_count = np.count_nonzero(window == Connect4.EMPTY)

    if ai_count == 4:
        score += 1000
    elif ai_count == 3 and empty_count == 1:
        score += 50
    elif ai_count == 2 and empty_count == 2:
        score += 10

    if opp_count == 3 and empty_count == 1:
        score -= 80
    elif opp_count == 2 and empty_count == 2:
        score -= 5

    return score


def evaluate(game, ai_player):
    """Evaluate the full board using the notebook's original heuristic."""
    board = game.board
    score = 0

    center_col = board[:, Connect4.COLS // 2]
    center_count = np.count_nonzero(center_col == ai_player)
    score += center_count * 3

    for row in range(Connect4.ROWS):
        for col in range(Connect4.COLS - 3):
            window = board[row, col : col + 4]
            score += evaluate_window(window, ai_player)

    for col in range(Connect4.COLS):
        for row in range(Connect4.ROWS - 3):
            window = board[row : row + 4, col]
            score += evaluate_window(window, ai_player)

    for row in range(Connect4.ROWS - 3):
        for col in range(Connect4.COLS - 3):
            window = np.array([board[row + i][col + i] for i in range(4)])
            score += evaluate_window(window, ai_player)

    for row in range(3, Connect4.ROWS):
        for col in range(Connect4.COLS - 3):
            window = np.array([board[row - i][col + i] for i in range(4)])
            score += evaluate_window(window, ai_player)

    return score
