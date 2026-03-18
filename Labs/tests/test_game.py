import numpy as np

from connect4.game import Connect4


def play_moves(moves):
    game = Connect4()
    for move in moves:
        game.drop_piece(move)
    return game


def test_actions_on_empty_board():
    game = Connect4()
    assert game.actions() == [0, 1, 2, 3, 4, 5, 6]


def test_drop_piece_returns_expected_row_and_updates_turn():
    game = Connect4()
    row = game.drop_piece(3)
    assert row == 5
    assert game.board[5][3] == Connect4.PLAYER1
    assert game.current_player == Connect4.PLAYER2


def test_undo_move_restores_board_and_current_player():
    game = Connect4()
    row = game.drop_piece(4)
    game.undo_move(4, row)
    assert game.board[5][4] == Connect4.EMPTY
    assert game.current_player == Connect4.PLAYER1


def test_copy_is_independent():
    game = Connect4()
    game.drop_piece(2)
    clone = game.copy()
    clone.drop_piece(3)
    assert not np.array_equal(game.board, clone.board)


def test_vertical_win_detection():
    game = play_moves([0, 1, 0, 1, 0, 1, 0])
    assert game.check_winner(Connect4.PLAYER1)
    assert game.is_terminal()


def test_horizontal_win_detection():
    game = play_moves([0, 0, 1, 1, 2, 2, 3])
    assert game.check_winner(Connect4.PLAYER1)
    assert game.is_terminal()


def test_positive_diagonal_win_detection():
    game = play_moves([0, 1, 1, 2, 3, 2, 2, 3, 6, 3, 3])
    assert game.check_winner(Connect4.PLAYER1)


def test_negative_diagonal_win_detection():
    game = play_moves([3, 2, 2, 1, 0, 1, 1, 0, 6, 0, 0])
    assert game.check_winner(Connect4.PLAYER1)


def test_terminal_score_win_loss_draw():
    win_game = play_moves([0, 1, 0, 1, 0, 1, 0])
    assert win_game.get_terminal_score(Connect4.PLAYER1) == 1000
    assert win_game.get_terminal_score(Connect4.PLAYER2) == -1000

    draw_board = [
        [2, 2, 1, 2, 1, 1, 2],
        [2, 1, 2, 1, 2, 1, 2],
        [1, 1, 2, 1, 2, 2, 2],
        [2, 2, 1, 2, 1, 1, 1],
        [1, 1, 1, 2, 2, 2, 1],
        [2, 1, 1, 1, 2, 1, 2],
    ]
    draw_game = Connect4(board=draw_board, current_player=Connect4.PLAYER1)
    assert draw_game.is_terminal()
    assert draw_game.get_terminal_score(Connect4.PLAYER1) == 0
