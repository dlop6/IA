import random

from .base import BaseAgent


class RandomAgent(BaseAgent):
    """Agent that picks a valid move uniformly at random."""

    def select_action(self, game):
        return random.choice(game.actions())
