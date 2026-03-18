import random
import time

from IPython.display import clear_output

from .agents.alphabeta import SmartAlphaBetaAgent
from .agents.random_agent import RandomAgent
from .game import Connect4
from .visualization import display_board, print_board


def play_match(
    player1_agent,
    player2_agent,
    *,
    verbose=False,
    visual=False,
    result_labels=None,
    move_labels=None,
    print_result=True,
):
    """Run a complete game and return a structured match summary."""
    game = Connect4()
    move_count = 0
    last_move = None
    moves = []
    node_totals = {Connect4.PLAYER1: 0, Connect4.PLAYER2: 0}

    if result_labels is None:
        result_labels = {
            Connect4.PLAYER1: "Player 1 gana.",
            Connect4.PLAYER2: "Player 2 gana.",
            "draw": "Empate.",
        }
    if move_labels is None:
        move_labels = {
            Connect4.PLAYER1: "Player 1",
            Connect4.PLAYER2: "Player 2",
        }

    while not game.is_terminal():
        agent = player1_agent if game.current_player == Connect4.PLAYER1 else player2_agent
        player = game.current_player

        if hasattr(agent, "get_best_move"):
            col, nodes = agent.get_best_move(game)
        else:
            col = agent.select_action(game)
            nodes = None

        row = game.drop_piece(col)
        last_move = (row, col)
        moves.append({"player": player, "column": col, "row": row})
        move_count += 1

        if nodes is not None:
            node_totals[player] += nodes

        if verbose:
            if player == Connect4.PLAYER1:
                print(f"{move_labels[player]} juega columna: {col}")
            elif nodes is not None:
                print(f"{move_labels[player]} juega columna: {col}  [nodos: {nodes:,}]")
            else:
                print(f"{move_labels[player]} juega columna: {col}")

            if visual:
                clear_output(wait=True)
                display_board(game, title=f"Movimiento #{move_count}", last_move=last_move)
            else:
                print_board(game)

    if game.check_winner(Connect4.PLAYER1):
        winner = Connect4.PLAYER1
        result = result_labels[Connect4.PLAYER1]
    elif game.check_winner(Connect4.PLAYER2):
        winner = Connect4.PLAYER2
        result = result_labels[Connect4.PLAYER2]
    else:
        winner = None
        result = result_labels["draw"]

    if verbose and visual:
        clear_output(wait=True)
        display_board(game, title=f"FINAL - {result}", last_move=last_move)

    if print_result:
        print(f"Resultado: {result}  |  Movimientos: {move_count}")

    return {
        "winner": winner,
        "result": result,
        "move_count": move_count,
        "last_move": last_move,
        "moves": moves,
        "board": game.board.copy(),
        "node_totals": node_totals,
    }


def play_ai_vs_random(ai_depth=6, verbose=True, visual=True, seed=None):
    """Replay the notebook's AI vs random demo using imported modules."""
    if seed is not None:
        random.seed(seed)

    match = play_match(
        RandomAgent(ai_player=Connect4.PLAYER1),
        SmartAlphaBetaAgent(ai_player=Connect4.PLAYER2, depth=ai_depth),
        verbose=verbose,
        visual=visual,
        result_labels={
            Connect4.PLAYER1: "Aleatorio (X) gana.",
            Connect4.PLAYER2: "IA (O) GANA!",
            "draw": "Empate.",
        },
        move_labels={
            Connect4.PLAYER1: "Aleatorio (X)",
            Connect4.PLAYER2: "IA (O)",
        },
        print_result=True,
    )
    return match["result"]


def run_ai_vs_random_series(num_games=10, ai_depth=6, seed=None):
    """Run the notebook's batch AI vs random experiment and return aggregate counts."""
    if seed is not None:
        random.seed(seed)

    wins_ia = 0
    wins_random = 0
    draws = 0

    print(f"Jugando {num_games} partidas IA (d={ai_depth}) vs Aleatorio...\n")
    for index in range(num_games):
        result = play_ai_vs_random(ai_depth=ai_depth, verbose=False, visual=False)
        print(f"  Partida {index + 1}: {result}")
        if "IA" in result:
            wins_ia += 1
        elif "Aleatorio" in result:
            wins_random += 1
        else:
            draws += 1

    print(f"\n{'=' * 40}")
    print(f"Resultados en {num_games} partidas:")
    print(f"  IA gana:        {wins_ia}/{num_games}")
    print(f"  Aleatorio gana: {wins_random}/{num_games}")
    print(f"  Empates:        {draws}/{num_games}")

    return {
        "IA": wins_ia,
        "Random": wins_random,
        "Draw": draws,
    }


def play_human_vs_ai(ai_depth=6, input_fn=input):
    """Interactive human vs AI demo preserved from the original notebook."""
    game = Connect4()
    ai_agent = SmartAlphaBetaAgent(ai_player=Connect4.PLAYER2, depth=ai_depth)
    last_move = None

    print("Connect Four! Tu eres ROJO (X), la IA es AMARILLO (O).")
    print("Ingresa el numero de columna (0-6).\n")
    display_board(game, title="Tu turno!", last_move=last_move)

    while not game.is_terminal():
        if game.current_player == Connect4.PLAYER1:
            valid = game.actions()
            while True:
                try:
                    col = int(input_fn(f"Tu turno (columnas validas {valid}): "))
                    if col in valid:
                        break
                    print(f"  Columna {col} no es valida. Intenta de nuevo.")
                except ValueError:
                    print("  Ingresa un numero entre 0 y 6.")

            row = game.drop_piece(col)
            last_move = (row, col)
            clear_output(wait=True)
            display_board(game, title=f"Tu (X) jugaste columna {col}", last_move=last_move)
        else:
            print("La IA esta pensando...")
            start = time.time()
            col, nodes = ai_agent.get_best_move(game)
            end = time.time()
            row = game.drop_piece(col)
            last_move = (row, col)
            clear_output(wait=True)
            display_board(
                game,
                title=f"IA (O) jugo columna {col}  [{nodes:,} nodos, {end - start:.2f}s]",
                last_move=last_move,
            )

    clear_output(wait=True)
    if game.check_winner(Connect4.PLAYER1):
        display_board(game, title="Tu ganas! Felicidades!", last_move=last_move)
    elif game.check_winner(Connect4.PLAYER2):
        display_board(game, title="La IA gana. Mejor suerte la proxima!", last_move=last_move)
    else:
        display_board(game, title="Empate!", last_move=last_move)
