import numpy as np


class Connect4:
    """State and rules for a Connect Four game."""

    ROWS = 6
    COLS = 7
    EMPTY = 0
    PLAYER1 = 1
    PLAYER2 = 2

    def __init__(self, board=None, current_player=PLAYER1):
        self.board = (
            np.array(board, dtype=int, copy=True)
            if board is not None
            else np.zeros((self.ROWS, self.COLS), dtype=int)
        )
        self.current_player = current_player

    def copy(self):
        """Return a deep copy of the game state."""
        return Connect4(board=self.board.copy(), current_player=self.current_player)

    def actions(self):
        """Return valid columns where a piece can be dropped."""
        return [col for col in range(self.COLS) if self.board[0][col] == self.EMPTY]

    def drop_piece(self, col):
        """
        Drop a piece for the current player and toggle turn.

        Returns the row where the piece landed, or -1 if the column is full.
        """
        for row in range(self.ROWS - 1, -1, -1):
            if self.board[row][col] == self.EMPTY:
                self.board[row][col] = self.current_player
                self.current_player = (
                    self.PLAYER2 if self.current_player == self.PLAYER1 else self.PLAYER1
                )
                return row
        return -1

    def undo_move(self, col, row):
        """Undo a move and restore the previous player turn."""
        self.board[row][col] = self.EMPTY
        self.current_player = (
            self.PLAYER2 if self.current_player == self.PLAYER1 else self.PLAYER1
        )

    def check_winner(self, player):
        """Return True if the requested player has 4 in a row."""
        b = self.board

        for row in range(self.ROWS):
            for col in range(self.COLS - 3):
                if b[row][col] == b[row][col + 1] == b[row][col + 2] == b[row][col + 3] == player:
                    return True

        for row in range(self.ROWS - 3):
            for col in range(self.COLS):
                if b[row][col] == b[row + 1][col] == b[row + 2][col] == b[row + 3][col] == player:
                    return True

        for row in range(self.ROWS - 3):
            for col in range(self.COLS - 3):
                if b[row][col] == b[row + 1][col + 1] == b[row + 2][col + 2] == b[row + 3][col + 3] == player:
                    return True

        for row in range(3, self.ROWS):
            for col in range(self.COLS - 3):
                if b[row][col] == b[row - 1][col + 1] == b[row - 2][col + 2] == b[row - 3][col + 3] == player:
                    return True

        return False

    def is_terminal(self):
        """Return True if the game is over."""
        return (
            self.check_winner(self.PLAYER1)
            or self.check_winner(self.PLAYER2)
            or len(self.actions()) == 0
        )

    def get_terminal_score(self, ai_player):
        """Return +1000 for win, -1000 for loss, and 0 for draw."""
        opponent = self.PLAYER1 if ai_player == self.PLAYER2 else self.PLAYER2
        if self.check_winner(ai_player):
            return 1000
        if self.check_winner(opponent):
            return -1000
        return 0
