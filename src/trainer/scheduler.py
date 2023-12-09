import numpy as np


def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)

    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)

    return _lr_adjuster


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
