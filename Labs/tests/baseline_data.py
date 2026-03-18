BASELINE_STATES = {
    "demo_minimax": {
        "moves": [3, 3, 4],
        "current_player": 2,
        "minimax": {"move": 2, "nodes": 2800},
        "alphabeta": {"move": 2, "nodes": 459},
        "smart_alphabeta": {"move": 2, "nodes": 8122},
    },
    "demo_compare": {
        "moves": [3, 2, 3, 4, 5],
        "current_player": 2,
        "minimax": {"move": 0, "nodes": 2800},
        "alphabeta": {"move": 0, "nodes": 190},
        "smart_alphabeta": {"move": 4, "nodes": 3337},
    },
    "midgame_1": {
        "moves": [3, 2, 3, 2, 4, 5, 4, 5],
        "current_player": 1,
        "minimax": {"move": 0, "nodes": 2716},
        "alphabeta": {"move": 0, "nodes": 294},
        "smart_alphabeta": {"move": 3, "nodes": 1902},
    },
}

SEEDED_AI_VS_RANDOM = {
    "seed": 123,
    "result": "IA (O) GANA!",
    "winner": 2,
    "move_count": 24,
    "columns": [0, 3, 0, 0, 3, 4, 0, 2, 6, 3, 3, 3, 4, 3, 2, 2, 1, 1, 2, 2, 2, 4, 1, 1],
    "final_board": [
        [0, 0, 1, 2, 0, 0, 0],
        [0, 0, 2, 2, 0, 0, 0],
        [1, 2, 1, 1, 0, 0, 0],
        [2, 1, 2, 2, 2, 0, 0],
        [1, 2, 1, 1, 1, 0, 0],
        [1, 1, 2, 2, 2, 0, 1],
    ],
}

SEEDED_AI_VS_RANDOM_SERIES = {
    "seed": 123,
    "num_games": 10,
    "counts": {"IA": 10, "Random": 0, "Draw": 0},
}
