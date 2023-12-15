import subprocess
from pathlib import Path

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


def start_subprocess(command, print_command=False):
    if print_command:
        print(" ".join(command) + "\n")

    process = subprocess.Popen(command)
    return process.wait()


if __name__ == "__main__":
    pretrained_path = "openai"
    wise_path = "openai"

    for i, dataset in enumerate(DATASET_SEQ):
        # training
        commands = [
            "python",
            "kd_train.py",
            "--options",
            f"data.name={dataset}",
            f"model.pretrained={pretrained_path}",
            f"model.wise_path={wise_path}",
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
                "--options",
                f"data.name={eval_dataset}",
                f"model.pretrained={pretrained_path}",
                f"model.wise_path={wise_path}",
            ]
            start_subprocess(commands, print_command=False)
