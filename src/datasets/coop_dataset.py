from src.datasets.base import BaseCoOPDataset


class Caltech101(BaseCoOPDataset):
    dataset_name = "caltech-101"
    annotation_filename = "split_zhou_Caltech101.json"


class OxfordPets(BaseCoOPDataset):
    dataset_name = "oxford-pets"
    annotation_filename = "split_zhou_OxfordPets.json"


class StanfordCars(BaseCoOPDataset):
    dataset_name = "stanford-cars"
    annotation_filename = "split_zhou_StanfordCars.json"


class Flowers102(BaseCoOPDataset):
    dataset_name = "flowers-102"
    annotation_filename = "split_zhou_OxfordFlowers.json"


class Food101(BaseCoOPDataset):
    dataset_name = "food-101"
    annotation_filename = "split_zhou_Food101.json"


class FGVCAircraft(BaseCoOPDataset):
    dataset_name = "fgvc-aircraft"
    annotation_filename = "annotations.json"


class EuroSAT(BaseCoOPDataset):
    dataset_name = "eurosat"
    annotation_filename = "split_zhou_EuroSAT.json"


class UCF101(BaseCoOPDataset):
    dataset_name = "ucf-101"
    annotation_filename = "split_zhou_UCF101.json"


if __name__ == "__main__":
    dataset = UCF101(root="/mnt/data/classification", mode="test")
    print(dataset[0])
    print(len(dataset))
    print(len(dataset.class_list))
