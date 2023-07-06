import math
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.nn.learning_rate_schedule import LearningRateSchedule


class SelfMadeDynamicLR(LearningRateSchedule):
    def __init__(self, learning_rate, decay_rate, decay_steps, warmup_steps, is_stair=True):
        super(SelfMadeDynamicLR, self).__init__()
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.is_stair = is_stair
        self.math_e = math.e
        self.warmup_steps = warmup_steps
        self.min = P.Minimum()
        self.pow = P.Pow()
        self.cast = P.Cast()

    def construct(self, global_step):
        p = self.cast(global_step, mstype.float32)
        if self.is_stair:
            p = P.FloorDiv()(p, self.decay_steps)
        else:
            p = p / self.decay_steps
        warmup_percent = self.cast(self.min(global_step, self.warmup_steps), mstype.float32) / self.warmup_steps
        return self.learning_rate * self.pow(self.math_e, -self.decay_rate * p) * warmup_percent
