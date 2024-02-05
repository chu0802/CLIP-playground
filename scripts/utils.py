import json
import subprocess
from ast import literal_eval
from pathlib import Path
from typing import List

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


class ContinualTrainer:
    def __init__(
        self,
        config_path: str = "configs/mix_teacher_config.yaml",
        training_dataset_seq: List[str] = DEFAULT_DATASET_SEQ,
        eval_dataset_seq: List[str] = DEFAULT_DATASET_SEQ,
        dump_results: bool = True,
    ):
        self.config_path = config_path
        self.training_dataset_seq = training_dataset_seq
        self.eval_dataset_seq = eval_dataset_seq
        self.dump_results = dump_results

        if self.dump_results:
            self.output_dir = Path("outputs") / Path(self.config_path).stem
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def aggregate_results(self):
        results_dict = dict()
        for dataset in self.training_dataset_seq:
            eval_result_path = get_output_dataset_dir(dataset) / "eval_results.json"

            with eval_result_path.open("r") as f:
                results = json.load(f)

            results_dict[dataset] = results

        return results_dict

    def format_results(self, res_dict, pad=4, decimal=2):
        longest_training_dataset_name_len = max(
            [len(k) for k in self.training_dataset_seq]
        )
        lines = []
        lines.append(
            (" " * pad).join(
                [" " * longest_training_dataset_name_len]
                + [
                    f"%{max(len(dataset), 5)}s" % (dataset)
                    for dataset in self.eval_dataset_seq
                ]
            )
        )

        for training_dataset in self.training_dataset_seq:
            line = [f"%{longest_training_dataset_name_len}s" % (training_dataset)]
            line += [
                f"%{len(eval_dataset)}s"
                % (f"{100*res_dict[training_dataset][eval_dataset]:.{decimal}f}")
                for eval_dataset in self.eval_dataset_seq
            ]
            lines.append((" " * pad).join(line))

        return "\n".join(lines) + "\n"

    def train_and_eval(self, pretrained_dataset=None, format=True):
        for training_dataset in self.training_dataset_seq:
            train_and_eval_script(
                config_path=self.config_path,
                training_dataset=training_dataset,
                pretrained_dataset=pretrained_dataset,
                eval_dataset_seq=self.eval_dataset_seq,
            )
            pretrained_dataset = training_dataset

        res = self.aggregate_results()

        if self.dump_results:
            with (self.output_dir / "final_results.json").open("w") as f:
                json.dump(res, f, indent=4)

        if format:
            formatted_results = self.format_results(res)
            with (self.output_dir / "formatted_results.txt").open("w") as f:
                f.write(formatted_results)
            print(formatted_results)

        return res


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


def train_and_eval_script(
    config_path: str = "configs/mix_teacher_config.yaml",
    training_module: str = "main.kd_train",
    training_dataset: str = "fgvc-aircraft",
    pretrained_dataset: str = None,
    eval_dataset_seq: List[str] = DEFAULT_DATASET_SEQ,
    **method_config,
):
    pretrained_model_path = get_model_path(pretrained_dataset)

    training_script(
        config_path=config_path,
        training_module=training_module,
        dataset=training_dataset,
        pretrained_model_path=pretrained_model_path,
        **method_config,
    )

    model_path = get_model_path(training_dataset)
    eval_results_path = get_output_dataset_dir(training_dataset) / "eval_results.json"

    eval_on_multiple_datasets_script(
        datasets=eval_dataset_seq,
        pretrained_model_path=model_path,
        dump_result_path=eval_results_path,
    )


def training_script(
    config_path,
    training_module="main.kd_train",
    dataset="fgvc-aircraft",
    pretrained_model_path="openai",
    sample_num=-1,
    max_epoch=10,
    **method_config,
):
    command = [
        "python",
        "-m",
        training_module,
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


def eval_on_multiple_datasets_script(
    config_path="configs/inference_config.yaml",
    eval_module="main.evaluate",
    datasets=DEFAULT_DATASET_SEQ,
    pretrained_model_path="openai",
    sample_num=-1,
    dump_result_path=None,
):
    eval_results = {}
    for eval_dataset in datasets:
        command = [
            "python",
            "-m",
            eval_module,
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
