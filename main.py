from omegaconf import OmegaConf

from src.datasets import Flowers102
from src.datasets.transform import load_model_transform
from src.models.clip import ClipModel


def main(config):
    model = ClipModel(config.model)
    _, eval_transform = load_model_transform(model.clip_base)

    dataset = Flowers102(config.data.root, mode="val", transform=eval_transform)

    image, label = dataset[-1]
    probs = model.inference(image.unsqueeze(0), dataset.class_list)

    print(
        f"""
        pred: {dataset.get_class_name(probs.argmax())} (prob.: {100*probs.max():0.2f}%)
        truth: {dataset.get_class_name(label)}

        {'Correct!' if probs.argmax() == label else 'Incorrect!'}
        """
    )


if __name__ == "__main__":
    config = OmegaConf.load("config.yaml")
    main(config)
