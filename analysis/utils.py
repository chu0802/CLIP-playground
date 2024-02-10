from omegaconf import OmegaConf

from src.datasets.transform import load_transform
from src.datasets.utils import build_iter_dataloader, get_dataloader
from src.models.clip import get_model

TEST_CONFIG = OmegaConf.create(
    {
        "model": {
            "vit_base": "ViT-B-16",
            "pretrained": "/home/chuyu/vllab/clip/outputs/ViT-B-16/fgvc-aircraft/latest/checkpoint_10.pth",
        },
        "data": {
            "name": "fgvc-aircraft",
            "root": "/mnt/data/classification",
        },
    }
)


def prepare_dataloader(
    dataset, batch_size=32, shuffle=True, drop_last=False, mode="train"
):
    train_trans, eval_trans = load_transform()
    config = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "drop_last": drop_last,
    }
    return get_dataloader(
        dataset,
        "/mnt/data/classification/",
        mode,
        train_trans if mode == "train" else eval_trans,
        **config
    )
