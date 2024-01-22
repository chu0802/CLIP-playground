from itertools import product
from pathlib import Path

from src.utils.subprocess import start_subprocess

OUTPUTS_ROOT = Path("outputs/ViT-B-16")
TRAIN_DATASET = "caltech-101"
TEST_DATASET = "fgvc-aircraft"


def main(args):
    pass


if __name__ == "__main__":
    ratio_prev_list = [3, 5, 7]
    threshold_list = [0.2, 0.25, 0.3]
    scale_list = [8, 10, 12]

    for ratio_prev, threshold, scale in product(
        ratio_prev_list, threshold_list, scale_list
    ):
        train_commands = [
            "python",
            "kd_train.py",
            "--cfg-path",
            "configs/mix_teacher_config.yaml",
            "--options",
            f"method.params.ratio_prev={ratio_prev}",
            f"method.params.threshold={threshold}",
            f"method.params.scale={scale}",
        ]

        start_subprocess(train_commands, print_command=True)

        eval_commands = [
            "python",
            "evaluate.py",
            "--cfg-path",
            "config.yaml",
            "--options",
            f"data.name={TEST_DATASET}",
            f"model.pretrained={OUTPUTS_ROOT.as_posix()}/{TRAIN_DATASET}/latest/checkpoint_10.pth",
        ]

        pipe_commands = [
            "tee",
            (OUTPUTS_ROOT / TRAIN_DATASET / "latest" / "log.txt").as_posix(),
        ]
        start_subprocess(eval_commands, print_command=True, pipe_command=pipe_commands)
