import json
from pathlib import Path

from torch.utils.data import DataLoader

from src.datasets import DATASET_MAPPING
from src.datasets.transform import load_transform


class DataIterativeLoader:
    def __init__(self, dataloader, device="cuda"):
        self.len = len(dataloader)
        self.dataloader = dataloader
        self.iterator = None
        self.device = device

    def init(self):
        self.iterator = iter(self.dataloader)

    def __next__(self):
        x, y = next(self.iterator)
        x = x.to(self.device)
        y = y.to(self.device)

        return x, y

    def __iter__(self):
        return self

    def __len__(self):
        return self.len


def build_dataloader(
    dataset,
    batch_size=8,
    num_workers=4,
    pin_memory=True,
    shuffle=False,
    drop_last=False,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
        drop_last=drop_last,
    )


def build_iter_dataloader(
    dataset,
    batch_size=8,
    num_workers=4,
    pin_memory=True,
    shuffle=False,
    drop_last=False,
    device="cuda",
    **kwargs,
):
    dataloader = build_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
        drop_last=drop_last,
    )

    return DataIterativeLoader(dataloader, device=device)


def get_dataloaders(config, device="cuda"):
    dataloaders = {}

    dataset_class = DATASET_MAPPING[config.data.name]
    train_transform, eval_transform = load_transform()

    for split_type, split_config in config.data.split.items():
        dataset = dataset_class(
            config.data.root,
            mode=split_config.split_name,
            transform=train_transform if split_type == "train" else eval_transform,
            sample_num=config.data.get("sample_num", -1),
        )

        dataloaders[split_type] = build_iter_dataloader(
            dataset, **split_config, device=device
        )

    return dataloaders


def load_class_name_list(config):
    dataset_class = DATASET_MAPPING[config.data.name]
    name, annotation_filename = (
        dataset_class.dataset_name,
        dataset_class.annotation_filename,
    )

    with (Path(config.data.root) / name / annotation_filename).open("r") as f:
        data = json.load(f)

    return data["class_names"]
