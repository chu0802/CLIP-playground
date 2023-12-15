from src.datasets.utils import get_dataloaders_from_config
from src.models.clip import get_model
from src.trainer import Trainer
from src.utils import get_config, setup_seeds, wandb_logger


@wandb_logger
def main(config):
    setup_seeds(config.task.seed)

    model = get_model(config, device="cuda")

    dataloaders = get_dataloaders_from_config(config)

    trainer = Trainer(model, dataloaders, config)

    trainer.logging(
        local_desc="zero shot", test_acc=trainer.evaluate(trainer.test_loader)
    )
    trainer.dump_results()


if __name__ == "__main__":
    config = get_config(mode="evaluate")
    main(config)
