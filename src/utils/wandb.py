import wandb
from src.utils.config import flatten_config


def wandb_logger(func):
    def wrap(config):
        wandb.init(
            project=config.data.name,
            name=config.data.name,
            config=flatten_config(config),
        )
        func(config)
        wandb.finish()

    return wrap
