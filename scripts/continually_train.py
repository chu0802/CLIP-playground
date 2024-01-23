import argparse
from pathlib import Path

from scripts import evaluation_script_on_multiple_datasets, training_script

DATASET_SEQ = [
    "fgvc-aircraft",
    "caltech-101",
    "dtd",
    "eurosat",
    "flowers-102",
    "oxford-pets",
    "stanford-cars",
    "ucf-101",
]

OUTPUTS_ROOT = Path("outputs/ViT-B-16")


def main(args):
    pretrained_path = "openai"

    for dataset in DATASET_SEQ:
        # training
        training_script(
            config_path=args.config_path,
            training_script="kd_train.py",
            dataset=dataset,
            pretrained_model_path=pretrained_path,
            sample_num=10,
            max_epoch=1,
        )

        # evaluation
        pretrained_path = OUTPUTS_ROOT / dataset / "latest" / "checkpoint_1.pth"
        eval_results_path = OUTPUTS_ROOT / dataset / "latest" / "eval_results.json"

        evaluation_script_on_multiple_datasets(
            pretrained_model_path=pretrained_path,
            sample_num=10,
            dump_result_path=eval_results_path,
        )

        break


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config_path", type=str, default="configs/mix_teacher_config.yaml")

    args = p.parse_args()

    main(args)
