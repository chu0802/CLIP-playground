from src.datasets.base import BaseClassificationDataset


class ImageNet(BaseClassificationDataset):
    dataset_name = "imagenet"
    annotation_filename = "annotations.json"


class Caltech101(BaseClassificationDataset):
    dataset_name = "caltech-101"
    annotation_filename = "split_zhou_Caltech101.json"


class OxfordPets(BaseClassificationDataset):
    dataset_name = "oxford-pets"
    annotation_filename = "split_zhou_OxfordPets.json"


class StanfordCars(BaseClassificationDataset):
    dataset_name = "stanford-cars"
    annotation_filename = "split_zhou_StanfordCars.json"


class Flowers102(BaseClassificationDataset):
    dataset_name = "flowers-102"
    annotation_filename = "split_zhou_OxfordFlowers.json"


class Food101(BaseClassificationDataset):
    dataset_name = "food-101"
    annotation_filename = "split_zhou_Food101.json"


class FGVCAircraft(BaseClassificationDataset):
    dataset_name = "fgvc-aircraft"
    annotation_filename = "annotations.json"


class EuroSAT(BaseClassificationDataset):
    dataset_name = "eurosat"
    annotation_filename = "split_zhou_EuroSAT.json"


class UCF101(BaseClassificationDataset):
    dataset_name = "ucf-101"
    annotation_filename = "split_zhou_UCF101.json"


if __name__ == "__main__":
    dataset = ImageNet(root="/mnt/data/classification", mode="val")
    print(dataset[0])
    print(len(dataset))
    print(len(dataset.class_list))
