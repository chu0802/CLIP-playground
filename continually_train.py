import argparse
import subprocess
from pathlib import Path

DATASET_SEQ = [
    "fgvc-aircraft",
    "caltech-101",
    "dtd",
    "flowers-102",
    "fgvc-aircraft",
    "caltech-101",
    "dtd",
    "flowers-102",
]
OUTPUTS_ROOT = Path("outputs/ViT-B-16")


def start_subprocess(command, print_command=False):
    if print_command:
        print(" ".join(command) + "\n")

    process = subprocess.Popen(command)
    return process.wait()


def main(args):
    pretrained_path = "openai"
    wise_path = "openai"

    for i, dataset in enumerate(DATASET_SEQ):
        # training
        commands = [
            "python",
            "kd_train.py",
            "--cfg-path",
            args.config_path,
            "--options",
            f"data.name={dataset}",
            f"model.pretrained={pretrained_path}",
            f"model.wise.path={wise_path}",
        ]
        start_subprocess(commands, print_command=True)

        if i > 0:
            wise_path = pretrained_path

        # evaluation
        pretrained_path = OUTPUTS_ROOT / dataset / "latest" / "checkpoint_10.pth"

        for eval_dataset in DATASET_SEQ:
            commands = [
                "python",
                "evaluate.py",
                "--cfg-path",
                args.config_path,
                "--options",
                f"data.name={eval_dataset}",
                f"model.pretrained={pretrained_path}",
                f"model.wise.path={wise_path}",
            ]
            start_subprocess(commands, print_command=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config_path", type=str, default="config/default_config.yaml")

    args = p.parse_args()

    main(args)
