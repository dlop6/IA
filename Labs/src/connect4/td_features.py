import numpy as np

from .game import Connect4


BOARD_FEATURE_LENGTH = Connect4.ROWS * Connect4.COLS
ACTION_FEATURE_LENGTH = Connect4.COLS
LANDING_ROW_FEATURE_LENGTH = Connect4.ROWS
TACTICAL_SCALAR_FEATURE_LENGTH = 15
STATE_ACTION_FEATURE_LENGTH = 1 + ACTION_FEATURE_LENGTH + LANDING_ROW_FEATURE_LENGTH + TACTICAL_SCALAR_FEATURE_LENGTH
TOTAL_WINDOWS = 69.0
ADJACENT_NEIGHBORS = 8.0
BOARD_CELLS = float(Connect4.ROWS * Connect4.COLS)
CENTER_COLUMNS = (2, 3, 4)
CENTER_CELLS = float(Connect4.ROWS * len(CENTER_COLUMNS))


def legal_actions(game):
    """Return the currently legal columns."""
    return game.actions()


def is_legal_action(game, action):
    """Return True if action is a valid playable column in the current state."""
    return action in legal_actions(game)


def encode_board(game):
    """
    Encode the board from the perspective of the player to move.

    This legacy helper is kept for compatibility; the TD agent now learns from the
    tactical state-action features returned by `encode_state_action`.
    """
    current = game.current_player
    opponent = Connect4.PLAYER1 if current == Connect4.PLAYER2 else Connect4.PLAYER2

    encoded = np.zeros_like(game.board, dtype=np.float64)
    encoded[game.board == current] = 1.0
    encoded[game.board == opponent] = -1.0
    return encoded.reshape(-1)


def encode_state_action(game, action):
    """
    Encode a tactical state-action feature vector for a linear Q(s, a) model.

    Feature order:
    - bias
    - action one-hot (7)
    - landing-row one-hot (6)
    - self_disc_fraction
    - opponent_disc_fraction
    - self_center_fraction
    - opponent_center_fraction
    - is_immediate_win
    - blocks_opponent_immediate_win
    - gives_opponent_immediate_win
    - self_open_twos_after_move
    - self_open_threes_after_move
    - opponent_open_twos_after_move
    - opponent_open_threes_after_move
    - self_immediate_winning_moves_next_turn
    - opponent_immediate_winning_moves_next_turn
    - adjacent_self_around_landing
    - adjacent_opponent_around_landing
    """
    row = _validate_and_get_landing_row(game, action)
    current_player = game.current_player
    opponent = _opponent_of(current_player)

    opponent_wins_before = set(_winning_actions_for_player(game, opponent))
    next_state = game.copy()
    next_state.drop_piece(action)

    action_one_hot = np.zeros(ACTION_FEATURE_LENGTH, dtype=np.float64)
    action_one_hot[action] = 1.0

    landing_row_one_hot = np.zeros(LANDING_ROW_FEATURE_LENGTH, dtype=np.float64)
    landing_row_one_hot[row] = 1.0

    next_board = next_state.board
    self_disc_fraction = float(np.count_nonzero(next_board == current_player) / BOARD_CELLS)
    opponent_disc_fraction = float(np.count_nonzero(next_board == opponent) / BOARD_CELLS)
    self_center_fraction = float(np.count_nonzero(next_board[:, CENTER_COLUMNS] == current_player) / CENTER_CELLS)
    opponent_center_fraction = float(np.count_nonzero(next_board[:, CENTER_COLUMNS] == opponent) / CENTER_CELLS)
    is_immediate_win = 1.0 if next_state.check_winner(current_player) else 0.0
    blocks_opponent_immediate_win = 1.0 if action in opponent_wins_before else 0.0

    self_wins_next = _winning_actions_for_player(next_state, current_player)
    opponent_wins_next = _winning_actions_for_player(next_state, opponent)

    scalar_features = np.array(
        [
            self_disc_fraction,
            opponent_disc_fraction,
            self_center_fraction,
            opponent_center_fraction,
            is_immediate_win,
            blocks_opponent_immediate_win,
            1.0 if opponent_wins_next else 0.0,
            _count_open_windows(next_state, current_player, target_count=2) / TOTAL_WINDOWS,
            _count_open_windows(next_state, current_player, target_count=3) / TOTAL_WINDOWS,
            _count_open_windows(next_state, opponent, target_count=2) / TOTAL_WINDOWS,
            _count_open_windows(next_state, opponent, target_count=3) / TOTAL_WINDOWS,
            len(self_wins_next) / ACTION_FEATURE_LENGTH,
            len(opponent_wins_next) / ACTION_FEATURE_LENGTH,
            *_adjacent_fractions(next_state, row, action, current_player, opponent),
        ],
        dtype=np.float64,
    )

    return np.concatenate(
        [
            np.array([1.0], dtype=np.float64),
            action_one_hot,
            landing_row_one_hot,
            scalar_features,
        ],
        dtype=np.float64,
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


def _winning_actions_for_player(game, player):
    winning_actions = []
    for action in legal_actions(game):
        simulated = Connect4(board=game.board, current_player=player)
        simulated.drop_piece(action)
        if simulated.check_winner(player):
            winning_actions.append(action)
    return winning_actions


def _count_open_windows(game, player, *, target_count):
    opponent = _opponent_of(player)
    count = 0
    for window in _iter_windows(game.board):
        player_count = int(np.count_nonzero(window == player))
        empty_count = int(np.count_nonzero(window == Connect4.EMPTY))
        opponent_count = int(np.count_nonzero(window == opponent))
        if player_count == target_count and empty_count == 4 - target_count and opponent_count == 0:
            count += 1
    return float(count)


def _iter_windows(board):
    for row in range(Connect4.ROWS):
        for col in range(Connect4.COLS - 3):
            yield board[row, col : col + 4]

    for row in range(Connect4.ROWS - 3):
        for col in range(Connect4.COLS):
            yield board[row : row + 4, col]

    for row in range(Connect4.ROWS - 3):
        for col in range(Connect4.COLS - 3):
            yield np.array([board[row + offset, col + offset] for offset in range(4)])

    for row in range(3, Connect4.ROWS):
        for col in range(Connect4.COLS - 3):
            yield np.array([board[row - offset, col + offset] for offset in range(4)])


def _adjacent_fractions(game, row, col, current_player, opponent):
    self_adjacent = 0
    opponent_adjacent = 0
    for row_offset in (-1, 0, 1):
        for col_offset in (-1, 0, 1):
            if row_offset == 0 and col_offset == 0:
                continue

            next_row = row + row_offset
            next_col = col + col_offset
            if 0 <= next_row < Connect4.ROWS and 0 <= next_col < Connect4.COLS:
                piece = game.board[next_row, next_col]
                if piece == current_player:
                    self_adjacent += 1
                elif piece == opponent:
                    opponent_adjacent += 1

    return (
        float(self_adjacent / ADJACENT_NEIGHBORS),
        float(opponent_adjacent / ADJACENT_NEIGHBORS),
    )


def _opponent_of(player):
    return Connect4.PLAYER1 if player == Connect4.PLAYER2 else Connect4.PLAYER2
