import math
from typing import Callable

import numpy as np

EPS = 1e-8


def warmup_cosine_decay_scheduler(
    init_value: float,
    peak_value: float,
    end_value: float,
    warmup_steps: int,
    decay_steps: int,
) -> Callable[[int], float]:
    def scheduler(step: int) -> float:
        if step < warmup_steps:
            # Warmup phase: linear interpolation from init to peak
            return init_value + (peak_value - init_value) * (step / warmup_steps)
        # NOTE: uses optax style (`decay_steps` = total schedule length)
        # https://github.com/google-deepmind/optax/blob/main/optax/schedules/_schedule.py#L652
        elif step < decay_steps:
            # Cosine decay phase
            decay_step = step - warmup_steps
            progress = decay_step / (decay_steps - warmup_steps)
            # Cosine decay from peak to end
            return end_value + (peak_value - end_value) * 0.5 * (1 + math.cos(math.pi * progress))
        else:
            return end_value

    return scheduler
