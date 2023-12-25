import torch
from tqdm import tqdm

from src.datasets.transform import load_transform
from src.datasets.utils import get_dataloader
from src.models.clip import ClipBase
from src.utils import setup_seeds

DATA_ROOT = "/mnt/data/classification"


def get_feature_distance(pre, fine, loader):
    loader.init()
    distance = []
    pre.eval()
    fine.eval()

    with torch.no_grad():
        for x, _ in tqdm(loader, total=len(loader)):
            fine_features = fine(x)
            pre_features = pre(x)

            distance.append(torch.norm(fine_features - pre_features, dim=-1))

    return torch.cat(distance)


if __name__ == "__main__":
    setup_seeds(1102)
    finetuned_model = ClipBase("ViT-B-16")
    state_dict = torch.load(
        "/home/chuyu/vllab/clip/outputs/ViT-B-16/fgvc-aircraft/latest/checkpoint_20.pth"
    )["model"]
    finetuned_model.model.visual.load_state_dict(state_dict)
    finetuned_model.to("cuda")

    pretrained_model = ClipBase("ViT-B-16", "openai").to("cuda")

    train_transform, _ = load_transform()

    finetuned_data_config = {
        "batch_size": 32,
        "shuffle": True,
        "drop_last": False,
    }

    imagenet_data_config = {
        "batch_size": 32,
        "shuffle": True,
        "drop_last": False,
        "sample_num": 10000,
    }

    finetuned_loader = get_dataloader(
        "fgvc-aircraft", DATA_ROOT, "train", train_transform, **finetuned_data_config
    )
    imagenet_loader = get_dataloader(
        "imagenet", DATA_ROOT, "train", train_transform, **imagenet_data_config
    )

    finetuned_data_distance = get_feature_distance(
        pretrained_model, finetuned_model, finetuned_loader
    )
    imagenet_data_distance = get_feature_distance(
        pretrained_model, finetuned_model, imagenet_loader
    )

    torch.save(finetuned_data_distance, "finetuned_data_distance.pt")
    torch.save(imagenet_data_distance, "imagenet_data_distance.pt")
