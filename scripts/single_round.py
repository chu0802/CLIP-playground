import argparse
from pathlib import Path

from scripts import evaluation_script_on_multiple_datasets, training_script

OUTPUTS_ROOT = Path("outputs/ViT-B-16")
PREV_DATASET = "fgvc-aircraft"


def main(args):
    pretrained_path = OUTPUTS_ROOT / PREV_DATASET / "latest" / "checkpoint_10.pth"

    training_script(
        config_path=args.config_path,
        training_script="kd_train.py",
        dataset=args.dataset,
        pretrained_model_path=pretrained_path,
    )

    # evaluation
    pretrained_path = OUTPUTS_ROOT / args.dataset / "latest" / "checkpoint_10.pth"
    eval_results_path = OUTPUTS_ROOT / args.dataset / "latest" / "eval_results.json"

    evaluation_script_on_multiple_datasets(
        pretrained_model_path=pretrained_path,
        dump_result_path=eval_results_path,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config_path", type=str, default="configs/mix_teacher_config.yaml")
    p.add_argument("--dataset", type=str, default="fgvc-aircraft")

    args = p.parse_args()

    main(args)
