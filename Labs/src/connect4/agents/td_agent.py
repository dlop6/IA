import random
from pathlib import Path

import numpy as np

from ..td_features import STATE_ACTION_FEATURE_LENGTH, encode_state_action, legal_actions
from .base import BaseAgent


class TDQLearningAgent(BaseAgent):
    """Linear Q-learning agent over a deterministic state-action feature vector."""

    def __init__(
        self,
        ai_player=2,
        *,
        learning_rate=0.1,
        discount=0.99,
        epsilon=0.1,
        weights=None,
        seed=None,
    ):
        super().__init__(ai_player=ai_player)
        self.learning_rate = float(learning_rate)
        self.discount = float(discount)
        self.epsilon = float(epsilon)
        self.rng = random.Random(seed)

        if weights is None:
            self.weights = np.zeros(STATE_ACTION_FEATURE_LENGTH, dtype=np.float32)
        else:
            self.weights = np.asarray(weights, dtype=np.float32)
            if self.weights.shape != (STATE_ACTION_FEATURE_LENGTH,):
                raise ValueError(
                    "Weights must have shape "
                    f"({STATE_ACTION_FEATURE_LENGTH},), got {self.weights.shape}."
                )

    def feature_vector(self, game, action):
        """Return the deterministic feature vector for a state-action pair."""
        return encode_state_action(game, action)

    def q_value(self, game, action):
        """Compute the linear action-value estimate Q(s, a) = w . x(s, a)."""
        return float(np.dot(self.weights, self.feature_vector(game, action)))

    def greedy_action(self, game):
        """Choose the legal action with the highest Q-value; ties break by lower column index."""
        actions = legal_actions(game)
        if not actions:
            raise ValueError("Cannot select an action from a terminal state with no legal actions.")

        q_values = [(action, self.q_value(game, action)) for action in actions]
        best_action, _ = max(q_values, key=lambda item: (item[1], -item[0]))
        return best_action

    def select_action(self, game, epsilon=None):
        """Choose an action using epsilon-greedy exploration over legal moves."""
        actions = legal_actions(game)
        if not actions:
            raise ValueError("Cannot select an action from a terminal state with no legal actions.")

        epsilon = self.epsilon if epsilon is None else float(epsilon)
        if self.rng.random() < epsilon:
            return self.rng.choice(actions)
        return self.greedy_action(game)

    def max_next_q_value(self, next_game):
        """Return the max Q-value over legal actions in the next state."""
        actions = legal_actions(next_game)
        if not actions:
            return 0.0
        return max(self.q_value(next_game, action) for action in actions)

    def update(
        self,
        game,
        action,
        reward,
        next_game,
        done,
        *,
        bootstrap_scale=1.0,
    ):
        """
        Apply one linear Q-learning update and return the TD error.

        `bootstrap_scale` is kept explicit so future self-play code can control how the
        next-state bootstrap term is interpreted in alternating-turn zero-sum settings.
        The default value 1.0 is the standard Q-learning target.
        """
        features = self.feature_vector(game, action)
        current_q = float(np.dot(self.weights, features))

        if done:
            target = float(reward)
        else:
            target = float(reward) + self.discount * float(bootstrap_scale) * self.max_next_q_value(next_game)

        td_error = target - current_q
        self.weights = self.weights + self.learning_rate * td_error * features
        return float(td_error)

    def save(self, path):
        """Persist weights and core hyperparameters to a .npz file."""
        path = Path(path)
        np.savez(
            path,
            weights=self.weights,
            learning_rate=np.array(self.learning_rate, dtype=np.float32),
            discount=np.array(self.discount, dtype=np.float32),
            epsilon=np.array(self.epsilon, dtype=np.float32),
        )

    @classmethod
    def load(cls, path, *, ai_player=2, seed=None):
        """Load a saved TD agent from disk."""
        path = Path(path)
        with np.load(path) as payload:
            return cls(
                ai_player=ai_player,
                learning_rate=float(payload["learning_rate"]),
                discount=float(payload["discount"]),
                epsilon=float(payload["epsilon"]),
                weights=payload["weights"],
                seed=seed,
            )
