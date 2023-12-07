from omegaconf import OmegaConf
from tqdm import tqdm

from src.datasets import Flowers102, ImageNet
from src.datasets.transform import load_model_transform
from src.datasets.utils import build_dataloader
from src.models.clip import ClipModel


def main(config):
    model = ClipModel(config.model)
    model.clip_base.cuda()
    model.clip_base.eval()

    _, eval_transform = load_model_transform(model.clip_base)

    dataset = ImageNet(
        config.data.root,
        mode="val",
        transform=eval_transform,
        sample_num=config.data.sample_num,
    )

    dataloader = build_dataloader(dataset, batch_size=64)

    correct, total = 0, 0
    for images, labels in tqdm(dataloader):
        images = images.cuda()
        labels = labels.cuda()

        probs = model.inference(images, dataset.class_list)

        preds = probs.argmax(dim=1)

        correct += sum(preds == labels).item()
        total += len(preds)

    print(f"Accuracy: {correct / total}")


if __name__ == "__main__":
    config = OmegaConf.load("config.yaml")
    main(config)
