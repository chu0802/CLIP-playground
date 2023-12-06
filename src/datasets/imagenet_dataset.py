from src.datasets.base import BaseClassificationDataset


class ImageNet(BaseClassificationDataset):
    dataset_name = "imagenet"

    def __init__(self, root, mode="train", transform=None):
        super().__init__(root, mode, transform, cache=True)
        self._class_id_to_name = self.build_class_id_to_name(
            path=self.root / "classnames.txt"
        )

    def _make_dataset(self):
        images_root = self.root / "images" / self.mode
        data_list = []
        class_list = sorted(images_root.iterdir())
        for class_id, path in enumerate(class_list):
            if not path.is_dir():
                raise ValueError(f"Invalid directory: {path}")
            for subpath in sorted(path.iterdir()):
                data_list.append((subpath.as_posix(), class_id))
        return data_list, [class_path.stem for class_path in class_list]

    def build_class_id_to_name(self, path):
        class_list = path.read_text().split("\n")[:-1]
        class_ids = [class_id.split()[0] for class_id in class_list]
        class_names = [class_id.split()[1] for class_id in class_list]

        return dict(zip(class_ids, class_names))

    def get_class_name(self, class_id_or_idx):
        if isinstance(class_id_or_idx, int):
            class_id = self.class_list[class_id_or_idx]
        return self._class_id_to_name[class_id]
