import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from src.datasets import ImageNet
from src.datasets.transform import load_transform
from src.datasets.utils import build_dataloader
from src.models.clip import load_model
from src.template import OPENAI_IMAGENET_TEMPLATE_LIST, SIMPLE_TEMPLATE_LIST


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
    dataloader = build_dataloader(dataset, batch_size=256)

    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.cuda()
            labels = labels.cuda()

            preds = model(images).argmax(dim=1)

            correct += sum(preds == labels).item()
            total += len(preds)

    print(f"Accuracy: {correct / total}")


if __name__ == "__main__":
    config = OmegaConf.load("config.yaml")
    main(config)
