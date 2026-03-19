import numpy as np

from .game import Connect4


BOARD_FEATURE_LENGTH = Connect4.ROWS * Connect4.COLS
ACTION_FEATURE_LENGTH = Connect4.COLS
LANDING_ROW_FEATURE_LENGTH = Connect4.ROWS
STATE_ACTION_FEATURE_LENGTH = (
    BOARD_FEATURE_LENGTH + ACTION_FEATURE_LENGTH + LANDING_ROW_FEATURE_LENGTH
)


def legal_actions(game):
    """Return the currently legal columns."""
    return game.actions()


def is_legal_action(game, action):
    """Return True if action is a valid playable column in the current state."""
    return action in legal_actions(game)


def encode_board(game):
    """
    Encode the board from the perspective of the player to move.

    Current-player pieces map to +1, opponent pieces map to -1, and empty cells map to 0.
    The returned vector is flattened row-major with fixed length 42.
    """
    current = game.current_player
    opponent = Connect4.PLAYER1 if current == Connect4.PLAYER2 else Connect4.PLAYER2

    encoded = np.zeros_like(game.board, dtype=np.float32)
    encoded[game.board == current] = 1.0
    encoded[game.board == opponent] = -1.0
    return encoded.reshape(-1)


def encode_state_action(game, action):
    """
    Encode a state-action pair for a linear Q(s, a) model.

    The vector concatenates:
    - perspective-normalized board features (42)
    - action one-hot vector over columns (7)
    - landing-row one-hot vector for the chosen legal action (6)
    """
    row = _validate_and_get_landing_row(game, action)

    action_one_hot = np.zeros(ACTION_FEATURE_LENGTH, dtype=np.float32)
    action_one_hot[action] = 1.0

    landing_row_one_hot = np.zeros(LANDING_ROW_FEATURE_LENGTH, dtype=np.float32)
    landing_row_one_hot[row] = 1.0

    return np.concatenate(
        [encode_board(game), action_one_hot, landing_row_one_hot],
        dtype=np.float32,
    )


def _validate_and_get_landing_row(game, action):
    """Validate action and return the row where the piece would land."""
    if not isinstance(action, (int, np.integer)):
        raise ValueError(f"Action must be an integer column index, got {type(action)!r}.")

    if action < 0 or action >= Connect4.COLS:
        raise ValueError(
            f"Action {action} is outside the valid column range 0-{Connect4.COLS - 1}."
        )

    if not is_legal_action(game, action):
        raise ValueError(f"Action {action} is not legal for the current state.")

    for row in range(Connect4.ROWS - 1, -1, -1):
        if game.board[row][action] == Connect4.EMPTY:
            return row

    raise ValueError(f"Action {action} has no landing row because the column is full.")
