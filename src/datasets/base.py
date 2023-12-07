import json
from abc import abstractmethod
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


def pil_loader(path: str):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class BaseClassificationDataset(Dataset):
    def __init__(self, root, mode="train", transform=None):
        self.root = Path(root) / self.dataset_name
        self.mode = mode
        self.data_list, self.class_list = self.make_dataset()
        self.transform = transform

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
        return self.class_list[class_idx]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        path, class_id = self.data_list[index]
        image = pil_loader(path)

        if self.transform:
            image = self.transform(image)

        return image, class_id
