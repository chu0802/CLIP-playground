import json
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def pil_loader(path: str):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class BaseClassificationDataset(Dataset):
    def __init__(self, root, mode="train", transform=None, sample_num=-1, seed=1102):
        self.root = Path(root) / self.dataset_name
        self.mode = mode
        self._data_list, self._class_name_list = self.make_dataset()
        self.transform = transform
        self.rng = np.random.default_rng(seed)

        if sample_num != -1:
            sample_idx = self.rng.choice(
                len(self._data_list), sample_num, replace=False
            )
            self._data_list = [self._data_list[i] for i in sample_idx]

    @property
    def class_name_list(self):
        return self._class_name_list

    def make_dataset(self):
        """
        data annotation format:
        {
            "data": {
                "train":[
                    [image_path, label],
                    ...
                ],
                "val": [
                    [image_path, label],
                    ...
                ],
                "test": [
                    [image_path, label],
                    ...
                ]
            },
            "class_names": [
                class_0_name,
                class_1_name,
                ...
            ]
        }
        """
        with (self.root / self.annotation_filename).open("r") as f:
            data = json.load(f)

        data_list = []
        for d in data["data"][self.mode]:
            data_list.append(((self.root / "images" / d[0]).as_posix(), d[1]))

        return data_list, data["class_names"]

    def get_class_name(self, class_idx):
        return self._class_name_list[class_idx]

    def __len__(self):
        return len(self._data_list)

    def __getitem__(self, index):
        path, label = self._data_list[index]
        image = pil_loader(path)

        if self.transform:
            image = self.transform(image)

        return image, label
