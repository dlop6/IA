from collections import deque


class TrainingStatsTracker:
    """Track rolling reward and outcome summaries for future training checkpoints."""

    def __init__(self, window_size=100):
        if window_size <= 0:
            raise ValueError("window_size must be greater than 0.")

        self.window_size = int(window_size)
        self.recent_rewards = deque(maxlen=self.window_size)
        self.total_episodes = 0

    def record_episode(self, reward):
        """Record a terminal episode reward from the learner's perspective."""
        self.recent_rewards.append(float(reward))
        self.total_episodes += 1

    def snapshot(self, *, episode, epsilon):
        """Return a serializable checkpoint summary."""
        rewards = list(self.recent_rewards)
        if rewards:
            avg_reward = sum(rewards) / len(rewards)
            wins = sum(1 for reward in rewards if reward > 0)
            losses = sum(1 for reward in rewards if reward < 0)
            draws = sum(1 for reward in rewards if reward == 0)
        else:
            avg_reward = 0.0
            wins = 0
            losses = 0
            draws = 0

        return {
            "episode": int(episode),
            "epsilon": float(epsilon),
            "window_size": len(rewards),
            "average_reward": float(avg_reward),
            "wins": wins,
            "losses": losses,
            "draws": draws,
        }
