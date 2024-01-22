from copy import deepcopy

from src.datasets.utils import get_dataloaders_from_config
from src.models.clip import get_model
from src.models.wise import wise_ft
from src.trainer import Trainer
from src.utils import get_config, local_logger, setup_seeds


@local_logger
def main(config):
    setup_seeds(config.task.seed)

    model = get_model(config, device="cuda")

    if config.model.wise.get("enable", False):
        pretrained_config = deepcopy(config)
        pretrained_config.model.pretrained = config.model.wise.path
        pretrained_model = get_model(pretrained_config, device="cuda")

        model = wise_ft(pretrained_model, model, alpha=config.model.wise.ratio)

        del pretrained_model

    dataloaders = get_dataloaders_from_config(config)

    trainer = Trainer(model, dataloaders, config)

    trainer.logging(
        local_desc="zero shot",
        test_acc=trainer.evaluate(trainer.test_loader),
        use_wandb=False,
    )
    trainer.dump_results(print_result=True)


if __name__ == "__main__":
    config = get_config(mode="evaluate")
    main(config)
