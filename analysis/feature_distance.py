import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from scripts.utils import DEFAULT_DATASET_SEQ, DEFAULT_OUTPUT_ROOT, get_model_path
from src.datasets import DATASET_MAPPING
from src.datasets.transform import load_transform
from src.datasets.utils import build_iter_dataloader, load_class_name_list
from src.models.clip import get_model


def main(config):

    # for dataset_name in DEFAULT_DATASET_SEQ:
    for dataset_name in ["imagenet"]:
        config.data.name = dataset_name

        class_name_list, _ = load_class_name_list(config)

        _, eval_transform = load_transform(config)

        data_config = {
            "batch_size": 256,
            "shuffle": True,
            "drop_last": False,
            "sample_num": 10_000 if dataset_name == "imagenet" else -1,
            "num_workers": 4,
            "pin_memory": True,
        }

        dataset = DATASET_MAPPING[dataset_name](
            config.data.root,
            mode="train",
            transform=eval_transform,
            sample_num=data_config["sample_num"],
            seed=1102,
        )

        dataloader = build_iter_dataloader(
            dataset,
            **data_config,
            device="cuda",
        )

        model = get_model(
            config,
            class_name_list,
            device="cuda",
            freeze=True,
            pretrained=False,
        )

        teacher_model = get_model(
            config,
            class_name_list,
            device="cuda",
            freeze=True,
            pretrained=True,
        )

        dataloader.init()

        student_feature_list = []
        teacher_feature_list = []
        indices = []
        for images, labels, idx in tqdm(dataloader):
            student_feature_list.append(model.encode(images=images))
            teacher_feature_list.append(teacher_model.encode(images=images))
            indices.append(idx)

        student_feature_list = torch.cat(student_feature_list, dim=0)
        teacher_feature_list = torch.cat(teacher_feature_list, dim=0)
        indices = torch.cat(indices, dim=0)

        data_distance = torch.norm(student_feature_list - teacher_feature_list, dim=1)
        print(f"Average distance for {dataset_name}: {data_distance.mean()}")

        argmax_idx = data_distance.argsort(descending=True)

        print(data_distance[argmax_idx[:100]])
        print(data_distance[argmax_idx[:100]].mean())

        for idx in argmax_idx[:100]:
            print(dataset._data_list[indices[idx]])


if __name__ == "__main__":
    order = 0
    training_dataset = "ucf-101"
    model_path = get_model_path(
        training_dataset, DEFAULT_OUTPUT_ROOT / f"order_{order}"
    )
    config = OmegaConf.create(
        {
            "model": {
                "vit_base": "ViT-B-16",
                "pretrained": model_path,
                "use_pure_clip": True,
            },
            "data": {
                "name": training_dataset,
                "root": "/mnt/data/classification",
            },
            "task": {},
        }
    )
    main(config)
