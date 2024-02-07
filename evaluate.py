from src.datasets.utils import get_dataloaders_from_config
from src.models.clip import get_model
from src.trainer import BaseTrainer as Trainer
from src.utils import get_config, setup_seeds


def main(config):
    setup_seeds(config.task.seed)

    model = get_model(config, device="cuda")

    dataloaders = get_dataloaders_from_config(config)

    trainer = Trainer(model, dataloaders, config, dump_result=False)

    trainer.logging(
        local_desc="zero shot",
        test_acc=trainer.evaluate(trainer.test_loader),
        use_wandb=False,
    )
    trainer.dump_results(print_result=True)


if __name__ == "__main__":
    config = get_config(mode="evaluate")
    main(config)
