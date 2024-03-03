from src.datasets import DATASET_MAPPING
from src.datasets.utils import get_conceptual_captions, get_dataloader, load_transform

from .base_trainer import BaseKDTrainer, BaseTrainer
from .mix_teacher_trainer import (
    MixTeacherKDTrainer,
    SplitTeacherKDTrainer,
    SplitTeacherPureClipKDTrainer,
)
from .we_trainer import get_weight_ensemble_trainer_class, get_wise_trainer_class
from .zscl_trainer import PreviousAwareZSCLTrainer, ZSCLTrainer

TRAINER_MAPPING = {
    "zscl": ZSCLTrainer,
    "previous_aware_zscl": PreviousAwareZSCLTrainer,
    "mix_teacher": MixTeacherKDTrainer,
    "split_teacher": SplitTeacherKDTrainer,
    "split_teacher_pure_clip": SplitTeacherPureClipKDTrainer,
}


def get_kd_trainer(model, dataloaders, config, teacher_models, job_id=None):
    if "ref_dataset" in config.method:
        train_transform, _ = load_transform(config)
        dataset_name, dataloader_config = (
            config.method.ref_dataset,
            config.method.ref_dataset_config,
        )

        dataloaders["ref"] = get_dataloader(
            dataset_name=dataset_name,
            root=config.data.root,
            mode=dataloader_config.split_name,
            transform=train_transform,
            seed=config.task.seed,
            distributed=config.task.get("distributed", False),
            **dataloader_config,
        )

    if config.method.name == "zscl":
        dataloaders["ref_sentences"] = get_conceptual_captions(
            config, size=config.method.ref_sentences_config.size
        )

    if config.method.name == "previous_aware_zscl":
        train_transform, _ = load_transform(config)

        previous_config = config.method.previous_config
        strategy = config.method.selected_strategy

        if strategy in DATASET_MAPPING:
            dataloader = get_dataloader(
                dataset_name=strategy,
                root=config.data.root,
                mode=previous_config.split_name,
                transform=train_transform,
                **previous_config,
            )

        dataloaders["prev"] = dataloader

    meta_trainer_class = TRAINER_MAPPING.get(config.method.name, BaseKDTrainer)

    if (
        config.method.get("weight_space_config", False)
        and config.method.weight_space_config.enable
    ):
        meta_trainer_class = get_weight_ensemble_trainer_class(meta_trainer_class)
    elif config.method.get("wise_config", False) and config.method.wise_config.enable:
        meta_trainer_class = get_wise_trainer_class(meta_trainer_class)

    return meta_trainer_class(model, dataloaders, config, teacher_models, job_id)
