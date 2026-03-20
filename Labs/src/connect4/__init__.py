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
from .task2_pipeline import (
    DEFAULT_TASK2_CONFIG,
    build_default_task2_epsilon_schedule,
    run_task2_pipeline,
)
from .task2_analysis import (
    export_task2_analysis_artifacts,
    extract_representative_matches,
    replay_match_states,
    summarize_task2_analysis,
)
from .task2_report import (
    create_task2_results_figure,
    export_task2_results_pdf,
    extract_task2_result_counts,
    load_task2_evaluation_summary,
)
from .training import run_self_play_episode, train_self_play
from .training_metrics import TrainingStatsTracker
from .visualization import board_to_text, create_board_figure, display_board, print_board

__all__ = [
    "ACTION_FEATURE_LENGTH",
    "AlphaBetaAgent",
    "BOARD_FEATURE_LENGTH",
    "Connect4",
    "ConstantEpsilonSchedule",
    "DEFAULT_TASK2_CONFIG",
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
    "build_default_task2_epsilon_schedule",
    "create_task2_results_figure",
    "create_board_figure",
    "display_board",
    "encode_board",
    "encode_state_action",
    "evaluate",
    "evaluate_window",
    "export_task2_analysis_artifacts",
    "export_task2_results_pdf",
    "extract_representative_matches",
    "extract_task2_result_counts",
    "is_legal_action",
    "legal_actions",
    "load_task2_evaluation_summary",
    "play_ai_vs_random",
    "play_human_vs_ai",
    "play_match",
    "print_board",
    "replay_match_states",
    "run_ai_vs_random_series",
    "run_self_play_episode",
    "run_task2_evaluation",
    "run_task2_match",
    "run_task2_pipeline",
    "summarize_task2_analysis",
    "train_self_play",
    "terminal_reward",
    "transition_reward",
]
