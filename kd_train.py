from copy import deepcopy

from src.datasets.utils import get_dataloaders
from src.models.clip import get_model
from src.trainer import KDTrainer
from src.utils import get_config, setup_seeds, wandb_logger


@wandb_logger
def main(config):
    setup_seeds(config.task.seed)

    model = get_model(config, device="cuda")

    teacher_config = deepcopy(config)

    teacher_config.model.pretrained = "openai"

    teacher_model = get_model(teacher_config, device="cuda")

    dataloaders = get_dataloaders(config)

    trainer = KDTrainer(model, dataloaders, config, teacher_model)

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
