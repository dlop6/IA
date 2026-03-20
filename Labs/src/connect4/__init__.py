from .agents import AlphaBetaAgent, MinimaxAgent, RandomAgent, SmartAlphaBetaAgent, TDQLearningAgent
from .evaluation import play_ai_vs_random, play_human_vs_ai, play_match, run_ai_vs_random_series
from .exploration import ConstantEpsilonSchedule, EpsilonSchedule, LinearDecayEpsilonSchedule
from .game import Connect4
from .heuristics import evaluate, evaluate_window
from .rewards import (
    DRAW_REWARD,
    LOSS_REWARD,
    NON_TERMINAL_REWARD,
    WIN_REWARD,
    terminal_reward,
    transition_reward,
)
from .td_features import (
    ACTION_FEATURE_LENGTH,
    BOARD_FEATURE_LENGTH,
    LANDING_ROW_FEATURE_LENGTH,
    STATE_ACTION_FEATURE_LENGTH,
    encode_board,
    encode_state_action,
    is_legal_action,
    legal_actions,
)
from .task2_eval import CONDITIONS, run_task2_evaluation, run_task2_match
from .training import run_self_play_episode, train_self_play
from .training_metrics import TrainingStatsTracker
from .visualization import board_to_text, display_board, print_board

__all__ = [
    "ACTION_FEATURE_LENGTH",
    "AlphaBetaAgent",
    "BOARD_FEATURE_LENGTH",
    "Connect4",
    "ConstantEpsilonSchedule",
    "LANDING_ROW_FEATURE_LENGTH",
    "DRAW_REWARD",
    "EpsilonSchedule",
    "LOSS_REWARD",
    "MinimaxAgent",
    "NON_TERMINAL_REWARD",
    "RandomAgent",
    "STATE_ACTION_FEATURE_LENGTH",
    "SmartAlphaBetaAgent",
    "TDQLearningAgent",
    "TrainingStatsTracker",
    "CONDITIONS",
    "LinearDecayEpsilonSchedule",
    "WIN_REWARD",
    "board_to_text",
    "display_board",
    "encode_board",
    "encode_state_action",
    "evaluate",
    "evaluate_window",
    "is_legal_action",
    "legal_actions",
    "play_ai_vs_random",
    "play_human_vs_ai",
    "play_match",
    "print_board",
    "run_ai_vs_random_series",
    "run_self_play_episode",
    "run_task2_evaluation",
    "run_task2_match",
    "train_self_play",
    "terminal_reward",
    "transition_reward",
]
