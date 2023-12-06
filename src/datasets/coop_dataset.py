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
