from copy import deepcopy

import torch

from src.datasets import DATASET_MAPPING
from src.datasets.base import NoisyImageListDataset
from src.datasets.utils import build_iter_dataloader, get_dataloader, load_transform
from src.utils import inference_feature_distance

from .base_trainer import BaseKDTrainer, BaseTrainer
from .mix_teacher_trainer import MixTeacherKDTrainer
from .zscl_trainer import PreviousAwareZSCLTrainer, ZSCLTrainer


def get_kd_trainer(model, dataloaders, config, teacher_models):
    if "ref_dataset" in config.method:
        train_transform, _ = load_transform()
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
            **dataloader_config,
        )

    if config.method.name == "previous_aware_zscl":
        train_transform, _ = load_transform()

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
        else:
            if strategy == "learnable_input":
                previous_approximation = (
                    torch.load("learnable_input_with_gt_labels.pt").cpu().detach()
                )
            elif strategy == "random_noise":
                previous_approximation = torch.randn(
                    previous_config.sample_num, 3, 224, 224
                )
            elif strategy == "from_ref_dataset":
                dist, indices = inference_feature_distance(
                    teacher_models["pretrained"],
                    teacher_models["prev"],
                    dataloaders["ref"],
                )
                argmax_idx = dist.argsort(descending=True)[: previous_config.sample_num]

                previous_approximation = deepcopy(dataloaders["ref"].dataloader.dataset)
                previous_approximation._data_list = [
                    previous_approximation._data_list[indices[idx]]
                    for idx in argmax_idx
                ]
            elif strategy == "pgd_attack":
                previous_approximation = NoisyImageListDataset(
                    noise_path="outputs/adv_images/adv_images.pt",
                    image_list_path="largest_distance.txt",
                    transform=train_transform,
                )
                previous_approximation._data_list = [
                    previous_approximation._data_list[i] for i in range(32)
                ]

            dataloader = build_iter_dataloader(
                previous_approximation,
                **previous_config,
            )

        dataloaders["prev"] = dataloader

    arguments = [model, dataloaders, config, teacher_models]

    match config.method.name:
        case "zscl":
            return ZSCLTrainer(*arguments)
        case "previous_aware_zscl":
            return PreviousAwareZSCLTrainer(*arguments)
        case "mix_teacher":
            return MixTeacherKDTrainer(*arguments)
        case _:
            return BaseKDTrainer(*arguments)
