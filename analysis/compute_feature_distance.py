from argparse import Namespace
from copy import deepcopy

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

import wandb
from analysis.utils import TEST_CONFIG, get_model, prepare_dataloader
from src.datasets import DATASET_MAPPING
from src.datasets.base import NoisyImageListDataset
from src.datasets.transform import RAW_TRANSFORM, load_transform
from src.datasets.utils import build_iter_dataloader, get_dataloader
from src.trainer.utils import CosineLRScheduler
from src.utils import inference_feature_distance, setup_seeds, wandb_logger

DATA_ROOT = "/mnt/data/classification"

DATASET_SEQ = [
    "fgvc-aircraft",
    "caltech-101",
    "dtd",
    "eurosat",
    "flowers-102",
    "oxford-pets",
    "stanford-cars",
    "ucf-101",
]


def main(config, seed=1102):
    setup_seeds(seed)

    train_transform, eval_transform = load_transform()

    imagenet_data_config = {
        "batch_size": 32,
        "shuffle": True,
        "drop_last": False,
        "sample_num": 10000,
    }

    dataset = DATASET_MAPPING["imagenet"](
        DATA_ROOT,
        mode="train",
        transform=train_transform,
        sample_num=imagenet_data_config["sample_num"],
        seed=seed,
    )

    dataloader = build_iter_dataloader(
        dataset,
        **imagenet_data_config,
        device="cuda",
    )

    # custom_config = {
    #     "batch_size": 32,
    #     "shuffle": True,
    #     "drop_last": False,
    #     "sample_num": -1,
    # }

    # dataset = NoisyImageListDataset(
    #     noise_path="outputs/adv_images/adv_images.pt",
    #     image_list_path="largest_distance.txt",
    #     transform=train_transform,
    # )
    # # dataset = ImageFolder(root="outputs/adv_images", transform=RAW_TRANSFORM)

    # dataloader = build_iter_dataloader(
    #     dataset,
    #     **custom_config,
    #     device="cuda",
    # )

    finetuned_model = get_model(config, device="cuda", freeze=True)
    pretrained_model = get_model(config, device="cuda", freeze=True, pretrained=True)

    imagenet_data_distance, indices = inference_feature_distance(
        pretrained_model, finetuned_model, dataloader
    )

    argmax_idx = imagenet_data_distance.argsort(descending=True)

    print(imagenet_data_distance[argmax_idx[:100]])
    print(imagenet_data_distance[argmax_idx[:100]].mean())
    print(imagenet_data_distance[argmax_idx[:32]].mean())

    for idx in argmax_idx[:100]:
        print(dataset._data_list[indices[idx]])


if __name__ == "__main__":
    config = OmegaConf.create(
        {
            "model": {
                "vit_base": "ViT-B-16",
                "pretrained": "/home/chuyu/vllab/clip/outputs/ViT-B-16/caltech-101/20240123082358/checkpoint_10.pth",
            },
            "data": {
                "name": "fgvc-aircraft",
                "root": "/mnt/data/classification",
            },
        }
    )

    main(config)
