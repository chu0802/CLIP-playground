from copy import deepcopy

from src.datasets.utils import get_dataloaders_from_config
from src.models.clip import get_model
from src.models.wise import wise_ft
from src.trainer import get_kd_trainer
from src.utils import get_config, setup_seeds, wandb_logger


@wandb_logger
def main(config):
    setup_seeds(config.task.seed)

    model = get_model(config, device="cuda")

    dataloaders = get_dataloaders_from_config(config)

    teachers = dict()

    teacher_config = deepcopy(config)

    teacher_config.model.pretrained = "openai"
    teachers["pretrained"] = get_model(teacher_config, device="cuda")

    if config.method.name in ["previous_aware_zscl", "mix_teacher", "split_teacher"]:
        prev_teacher_config = deepcopy(config)
        # to derive fine-tuned knowledge from teacher, we should not use pre-trained model as the teacher model.
        teachers["prev"] = get_model(prev_teacher_config, device="cuda")

    trainer = get_kd_trainer(model, dataloaders, config, teachers)

    # trainer.logging(
    #     local_desc="zero shot", test_acc=trainer.evaluate(trainer.test_loader)
    # )

    # validation provides nearly no information in my experiments
    trainer.train(set_validation=False)

    trainer.logging(
        local_desc="fine-tuned", test_acc=trainer.evaluate(trainer.test_loader)
    )
    trainer.dump_results()


if __name__ == "__main__":
    config = get_config(mode="train")
    main(config)
