import numpy as np


class CosineLRScheduler(object):
    def __init__(self, optimizer, task_config, num_steps):
        self.current_step = 0
        self.optimizer = optimizer

        init_lrs = task_config.init_lrs
        self.init_lrs = (
            init_lrs
            if isinstance(init_lrs, list)
            else [init_lrs for _ in optimizer.param_groups]
        )

        self.warmup_length = task_config.warmup_length
        self.num_steps = num_steps
        self._current_lr = self.init_lrs[0]

    def step(self):
        for param_group, init_lr in zip(self.optimizer.param_groups, self.init_lrs):
            if self.current_step < self.warmup_length:
                param_group["lr"] = (
                    init_lr * (self.current_step + 1) / self.warmup_length
                )
            else:
                e = self.current_step - self.warmup_length
                es = self.num_steps - self.warmup_length
                param_group["lr"] = 0.5 * (1 + np.cos(np.pi * e / es)) * init_lr

        self.current_step += 1
        self._current_lr = self.optimizer.param_groups[0]["lr"]

    def refresh(self):
        self.current_step = 0

    @property
    def current_lr(self):
        return self._current_lr
