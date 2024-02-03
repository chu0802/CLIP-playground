import json
import subprocess
from ast import literal_eval
from pathlib import Path

DEFAULT_OUTPUT_ROOT = Path("outputs/ViT-B-16")

DEFAULT_DATASET_SEQ = [
    "fgvc-aircraft",
    "caltech-101",
    "dtd",
    "eurosat",
    "flowers-102",
    "oxford-pets",
    "stanford-cars",
    "ucf-101",
]


def get_output_dataset_dir(
    dataset, output_root=DEFAULT_OUTPUT_ROOT, timestamp="latest"
):
    return output_root / dataset / timestamp


def get_model_path(
    dataset=None, output_root=DEFAULT_OUTPUT_ROOT, timestamp="latest", epoch=10
):
    if dataset is None:
        return "openai"
    model_dir = get_output_dataset_dir(dataset, output_root, timestamp)
    return model_dir / f"checkpoint_{epoch}.pth"


def start_subprocess(command, print_command=False, pipe_command=None):
    if print_command:
        print(" ".join(command) + "\n")

    if pipe_command:
        process = subprocess.Popen(command, stdout=subprocess.PIPE)
        output = subprocess.check_output(pipe_command, stdin=process.stdout)
        process.wait()
    else:
        output = subprocess.check_output(command)

    return output.decode("utf-8")


def training_script(
    config_path,
    training_script="kd_train.py",
    dataset="fgvc-aircraft",
    pretrained_model_path="openai",
    sample_num=-1,
    max_epoch=10,
    **method_config,
):
    command = [
        "python",
        training_script,
        "--cfg-path",
        config_path,
        "--options",
        f"data.name={dataset}",
        f"model.pretrained={pretrained_model_path}",
        f"data.sample_num={sample_num}",
        f"task.max_epoch={max_epoch}",
    ]

    if len(method_config) > 0:
        command += [f"method.{k}={v}" for k, v in method_config.items()]

    start_subprocess(command, print_command=True)


def evaluation_script_on_multiple_datasets(
    config_path="configs/inference_config.yaml",
    datasets=DEFAULT_DATASET_SEQ,
    pretrained_model_path="openai",
    sample_num=-1,
    dump_result_path=None,
):
    eval_results = {}
    for eval_dataset in datasets:
        command = [
            "python",
            "evaluate.py",
            "--cfg-path",
            config_path,
            "--options",
            f"model.pretrained={pretrained_model_path}",
            f"data.name={eval_dataset}",
            f"data.sample_num={sample_num}",
        ]

        res = start_subprocess(command, print_command=True)

        eval_results[eval_dataset] = float(literal_eval(res)["zero shot"]["test_acc"])

    if dump_result_path:
        with open(dump_result_path, "w") as f:
            json.dump(eval_results, f, indent=4)

    return eval_results
