from .game import Connect4


WIN_REWARD = 1.0
LOSS_REWARD = -1.0
DRAW_REWARD = 0.0
NON_TERMINAL_REWARD = 0.0


def terminal_reward(game, player):
    """Return the sparse terminal reward from the perspective of `player`."""
    if not game.is_terminal():
        return NON_TERMINAL_REWARD

    opponent = Connect4.PLAYER1 if player == Connect4.PLAYER2 else Connect4.PLAYER2
    if game.check_winner(player):
        return WIN_REWARD
    if game.check_winner(opponent):
        return LOSS_REWARD
    return DRAW_REWARD


def transition_reward(next_game, player):
    """
    Return the sparse reward after transitioning into `next_game`.

    Non-terminal states produce 0. Terminal states produce +1 / -1 / 0 depending on
    outcome from the perspective of `player`.
    """
    return terminal_reward(next_game, player)
