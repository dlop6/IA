import numpy as np
import pytest

from connect4 import (
    BOARD_FEATURE_LENGTH,
    Connect4,
    STATE_ACTION_FEATURE_LENGTH,
    encode_board,
    encode_state_action,
    is_legal_action,
    legal_actions,
)


BIAS_INDEX = 0
ACTION_START = 1
LANDING_START = ACTION_START + Connect4.COLS
SCALAR_START = LANDING_START + Connect4.ROWS


def play_moves(moves, current_player=Connect4.PLAYER1):
    game = Connect4(current_player=current_player)
    for move in moves:
        game.drop_piece(move)
    return game


def swap_players(board):
    swapped = np.array(board, copy=True)
    swapped[board == Connect4.PLAYER1] = Connect4.PLAYER2
    swapped[board == Connect4.PLAYER2] = Connect4.PLAYER1
    return swapped


def scalar_slice(features):
    return features[SCALAR_START:]


def test_encode_board_has_fixed_length_and_dtype():
    game = play_moves([3, 2, 3, 4, 5])
    encoded = encode_board(game)

    assert encoded.shape == (BOARD_FEATURE_LENGTH,)
    assert encoded.dtype == np.float64


def test_encode_state_action_has_fixed_length_and_is_deterministic():
    game = play_moves([3, 2, 3, 4, 5])

    first = encode_state_action(game, 0)
    second = encode_state_action(game, 0)

    assert first.shape == (STATE_ACTION_FEATURE_LENGTH,)
    assert first.dtype == np.float64
    assert np.array_equal(first, second)


def test_legal_action_helpers_match_game_actions():
    game = play_moves([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    assert legal_actions(game) == game.actions()
    assert is_legal_action(game, 0)
    assert is_legal_action(game, 6)

    game.drop_piece(0)
    assert not is_legal_action(game, 0)


def test_encode_state_action_rejects_illegal_actions():
    game = play_moves([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    with pytest.raises(ValueError, match="not legal"):
        encode_state_action(game, 0)

    with pytest.raises(ValueError, match="outside the valid column range"):
        encode_state_action(game, 7)

    with pytest.raises(ValueError, match="must be an integer"):
        encode_state_action(game, "3")


def test_perspective_normalization_matches_swapped_equivalent_state():
    original = play_moves([3, 2, 3, 4, 5, 4])
    mirrored = Connect4(
        board=swap_players(original.board),
        current_player=Connect4.PLAYER1 if original.current_player == Connect4.PLAYER2 else Connect4.PLAYER2,
    )

    assert np.array_equal(encode_board(original), encode_board(mirrored))
    assert np.array_equal(encode_state_action(original, 0), encode_state_action(mirrored, 0))


def test_state_action_encoding_has_expected_layout():
    game = play_moves([3, 2, 3, 4])
    encoded = encode_state_action(game, 3)

    assert encoded[BIAS_INDEX] == 1.0
    assert encoded[ACTION_START : ACTION_START + Connect4.COLS].sum() == 1.0
    assert encoded[ACTION_START + 3] == 1.0
    assert encoded[LANDING_START : LANDING_START + Connect4.ROWS].sum() == 1.0
    assert np.all((scalar_slice(encoded) >= 0.0) & (scalar_slice(encoded) <= 1.0))


def test_immediate_win_and_open_three_features_activate():
    game = play_moves([0, 6, 1, 6, 2, 5])
    encoded = encode_state_action(game, 3)
    scalars = scalar_slice(encoded)

    assert scalars[4] == 1.0
    assert scalars[8] > 0.0
    assert scalars[11] >= 0.0


def test_blocking_and_gives_opponent_win_features_differ_by_action():
    game = play_moves([6, 0, 6, 1, 5, 2])

    block = scalar_slice(encode_state_action(game, 3))
    ignore = scalar_slice(encode_state_action(game, 4))

    assert block[5] == 1.0
    assert block[6] == 0.0
    assert ignore[5] == 0.0
    assert ignore[6] == 1.0


def test_adjacent_features_capture_neighbor_mix():
    game = play_moves([3, 2, 4, 2])
    scalars = scalar_slice(encode_state_action(game, 3))

    assert np.isclose(scalars[13], 0.25)
    assert np.isclose(scalars[14], 0.25)
