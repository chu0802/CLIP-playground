import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from src.datasets import ImageNet
from src.datasets.transform import load_transform
from src.datasets.utils import build_iter_dataloader
from src.models.clip import load_model
from src.template import OPENAI_IMAGENET_TEMPLATE_LIST, SIMPLE_TEMPLATE_LIST
from src.utils.metrics import AccuracyMeter


def main(config):
    _, eval_transform = load_transform()

    dataset = ImageNet(
        config.data.root,
        mode="val",
        transform=eval_transform,
        sample_num=config.data.sample_num,
    )

    model = load_model(
        config.model,
        dataset.class_name_list,
        template_list=SIMPLE_TEMPLATE_LIST,
        device="cuda",
    ).eval()

    iter_dataloader = build_iter_dataloader(dataset, batch_size=256, device="cuda")
    scores = AccuracyMeter()

    with torch.no_grad():
        for images, labels in tqdm(iter_dataloader):
            preds = model(images).argmax(dim=1)
            scores += preds == labels

    print(f"Accuracy: {scores.acc()}")


if __name__ == "__main__":
    config = OmegaConf.load("config.yaml")
    main(config)
