class BaseAgent:
    """Shared agent interface for future search and TD agents."""

    def __init__(self, ai_player):
        self.ai_player = ai_player
        self.nodes_visited = 0

    @property
    def opponent(self):
        return 1 if self.ai_player == 2 else 2

    def select_action(self, game):
        raise NotImplementedError
