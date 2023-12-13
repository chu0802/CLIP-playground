from src.datasets.utils import get_dataloaders, load_class_name_list
from src.models.clip import get_model
from src.template import SIMPLE_TEMPLATE_LIST
from src.trainer import Trainer
from src.utils import get_config, setup_seeds, wandb_logger


@wandb_logger
def main(config):
    setup_seeds(config.task.seed)

    model = get_model(config, device="cuda")

    dataloaders = get_dataloaders(config)

    trainer = Trainer(model, dataloaders, config)

    trainer.logging(
        local_desc="zero shot", test_acc=trainer.evaluate(trainer.test_loader)
    )
    trainer.dump_results()


if __name__ == "__main__":
    config = get_config(mode="evaluate")
    main(config)
