from .core_dataset import (
    DTD,
    UCF101,
    Caltech101,
    EuroSAT,
    FGVCAircraft,
    Flowers102,
    Food101,
    ImageNet,
    OxfordPets,
    StanfordCars,
    SynthAirplane,
    SynthFGVCAircraft,
)

DATASET_MAPPING = {
    "dtd": DTD,
    "ucf-101": UCF101,
    "caltech-101": Caltech101,
    "eurosat": EuroSAT,
    "fgvc-aircraft": FGVCAircraft,
    "flowers-102": Flowers102,
    "food-101": Food101,
    "imagenet": ImageNet,
    "oxford-pets": OxfordPets,
    "stanford-cars": StanfordCars,
    "synth-fgvc-aircraft": SynthFGVCAircraft,
    "synth-airplane": SynthAirplane,
}
