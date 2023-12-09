import logging

import torch


def get_optimizer(model, task_config):
    optim_params = model.get_params()

    num_parameters = 0
    for param_group in optim_params:
        for p in param_group["params"]:
            num_parameters += p.data.nelement()
    logging.info(f"number of trainable parameters: {num_parameters}")

    return torch.optim.AdamW(
        optim_params,
        weight_decay=float(task_config.weight_decay),
    )
