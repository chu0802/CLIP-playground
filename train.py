from src.datasets.utils import get_dataloaders, load_class_name_list
from src.models.clip import load_model
from src.template import SIMPLE_TEMPLATE_LIST
from src.trainer import Trainer
from src.utils import get_config, setup_seeds, wandb_logger


@wandb_logger
def main(config):
    setup_seeds(config.task.seed)

    class_name_list = load_class_name_list(config)

    model = load_model(
        config.model,
        class_name_list,
        template_list=SIMPLE_TEMPLATE_LIST,
        freeze_classification_head=True,
        device="cuda",
    )

    dataloaders = get_dataloaders(config)

    trainer = Trainer(model, dataloaders, config)

    trainer.logging(
        local_desc="zero shot", test_acc=trainer.evaluate(trainer.test_loader)
    )

    trainer.train(set_validation=True)

    trainer.logging(
        local_desc="fine-tuned", test_acc=trainer.evaluate(trainer.test_loader)
    )
    trainer.dump_results()


if __name__ == "__main__":
    config = get_config(mode="train")
    main(config)
