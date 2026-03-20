from dataclasses import dataclass


class EpsilonSchedule:
    """Protocol-like base for epsilon schedules."""

    def value_at(self, step):
        raise NotImplementedError


@dataclass(frozen=True)
class ConstantEpsilonSchedule(EpsilonSchedule):
    """Return a fixed epsilon for every step."""

    epsilon: float

    def value_at(self, step):
        return float(self.epsilon)


@dataclass(frozen=True)
class LinearDecayEpsilonSchedule(EpsilonSchedule):
    """Linearly decay epsilon from `start` to `end` across `decay_steps`."""

    start: float
    end: float
    decay_steps: int

    def __post_init__(self):
        if self.decay_steps <= 0:
            raise ValueError("decay_steps must be greater than 0.")

    def value_at(self, step):
        if step <= 0:
            return float(self.start)
        if step >= self.decay_steps:
            return float(self.end)

        fraction = step / self.decay_steps
        return float(self.start + fraction * (self.end - self.start))
