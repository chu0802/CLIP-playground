import torch
from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from open_clip.transform import PreprocessCfg, image_transform_v2
from torchvision.transforms import CenterCrop, Compose, ConvertImageDtype, PILToTensor

DEFAULT_PREPROCESS_CONFIG = {
    "size": (224, 224),
    "mode": "RGB",
    "mean": OPENAI_DATASET_MEAN,
    "std": OPENAI_DATASET_STD,
    "interpolation": "bicubic",
    "resize_mode": "shortest",
    "fill_color": 0,
}


RAW_TRANSFORM = Compose(
    [
        CenterCrop(224),
        PILToTensor(),
        ConvertImageDtype(torch.float),
    ]
)


def load_transform(model_preprocess_config=None):
    if model_preprocess_config is None:
        model_preprocess_config = DEFAULT_PREPROCESS_CONFIG

    pp_cfg = PreprocessCfg(**model_preprocess_config)

    train_transform = image_transform_v2(
        pp_cfg,
        is_train=True,
    )

    eval_transform = image_transform_v2(
        pp_cfg,
        is_train=False,
    )

    return train_transform, eval_transform
