import matplotlib.pyplot as plt
import matplotlib.patches as patches

from .game import Connect4


def board_to_text(game):
    """Return the board rendered as text."""
    symbols = {
        Connect4.EMPTY: ".",
        Connect4.PLAYER1: "X",
        Connect4.PLAYER2: "O",
    }
    rows = [" ".join(symbols[cell] for cell in row) for row in game.board]
    rows.append(" ".join(str(i) for i in range(game.COLS)))
    return "\n".join([""] + rows + [""])


def print_board(game):
    """Print the board to stdout in the notebook's original text format."""
    print(board_to_text(game))


def display_board(game, title="", last_move=None):
    """Render the board in the same style as the original notebook."""
    colors = {
        Connect4.EMPTY: "white",
        Connect4.PLAYER1: "#FF4444",
        Connect4.PLAYER2: "#FFD700",
    }

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    ax.set_facecolor("#2856A3")
    fig.patch.set_facecolor("#1a1a2e")

    for row in range(game.ROWS):
        for col in range(game.COLS):
            piece = game.board[row][col]
            color = colors[piece]
            edge_color = "#00FF00" if last_move and last_move == (row, col) else "#1a3a6a"
            edge_width = 3 if last_move and last_move == (row, col) else 1.5
            circle = plt.Circle(
                (col, game.ROWS - 1 - row),
                0.4,
                facecolor=color,
                edgecolor=edge_color,
                linewidth=edge_width,
            )
            ax.add_patch(circle)

    ax.set_xlim(-0.5, game.COLS - 0.5)
    ax.set_ylim(-0.5, game.ROWS - 0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(game.COLS))
    ax.set_xticklabels(range(game.COLS), fontsize=14, fontweight="bold", color="white")
    ax.set_yticks([])
    ax.tick_params(axis="x", colors="white", length=0)

    legend_elements = [
        patches.Patch(facecolor="#FF4444", edgecolor="gray", label="Player 1 (X)"),
        patches.Patch(facecolor="#FFD700", edgecolor="gray", label="Player 2 (O)"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=9,
        facecolor="#1a1a2e",
        edgecolor="white",
        labelcolor="white",
    )

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", color="white", pad=10)

    plt.tight_layout()
    plt.show()
