from connect4 import AlphaBetaAgent, Connect4, MinimaxAgent, SmartAlphaBetaAgent

from tests.baseline_data import BASELINE_STATES


def make_game(moves):
    game = Connect4()
    for move in moves:
        game.drop_piece(move)
    return game


def test_minimax_alphabeta_and_smart_agent_match_baseline():
    for fixture in BASELINE_STATES.values():
        game = make_game(fixture["moves"])
        assert game.current_player == fixture["current_player"]

        minimax_move, minimax_nodes = MinimaxAgent(
            ai_player=game.current_player,
            depth=4,
        ).get_best_move(game.copy())
        assert minimax_move == fixture["minimax"]["move"]
        assert minimax_nodes == fixture["minimax"]["nodes"]

        alphabeta_move, alphabeta_nodes = AlphaBetaAgent(
            ai_player=game.current_player,
            depth=4,
        ).get_best_move(game.copy())
        assert alphabeta_move == fixture["alphabeta"]["move"]
        assert alphabeta_nodes == fixture["alphabeta"]["nodes"]

        smart_move, smart_nodes = SmartAlphaBetaAgent(
            ai_player=game.current_player,
            depth=6,
        ).get_best_move(game.copy())
        assert smart_move == fixture["smart_alphabeta"]["move"]
        assert smart_nodes == fixture["smart_alphabeta"]["nodes"]


def test_alphabeta_matches_minimax_move_and_uses_no_more_nodes():
    fixture = BASELINE_STATES["midgame_1"]
    game = make_game(fixture["moves"])
    minimax_move, minimax_nodes = MinimaxAgent(ai_player=game.current_player, depth=4).get_best_move(game.copy())
    alphabeta_move, alphabeta_nodes = AlphaBetaAgent(ai_player=game.current_player, depth=4).get_best_move(game.copy())

    assert alphabeta_move == minimax_move
    assert alphabeta_nodes <= minimax_nodes


def test_select_action_wrapper_matches_get_best_move():
    fixture = BASELINE_STATES["demo_compare"]
    game = make_game(fixture["moves"])
    agent = SmartAlphaBetaAgent(ai_player=game.current_player, depth=6)
    move, _ = agent.get_best_move(game.copy())
    assert agent.select_action(game.copy()) == move
