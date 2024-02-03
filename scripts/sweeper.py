import argparse
from itertools import product

from scripts import evaluation_script_on_multiple_datasets, training_script
from scripts.utils import get_model_path, get_output_dataset_dir


def main(args):
    params_list_dict = {
        "ratio_mix": [10],
        "normalize": [False],
    }
    pretrained_path = get_model_path(args.pretrained_dataset)

    for params in product(*params_list_dict.values()):
        params_dict = dict(zip(params_list_dict.keys(), params))

        training_script(
            config_path=args.config_path,
            training_script="kd_train.py",
            dataset=args.dataset,
            pretrained_model_path=pretrained_path,
            **params_dict,
        )

        model_path = get_model_path(args.dataset)
        eval_result_path = get_output_dataset_dir(args.dataset) / "eval_results.json"

        evaluation_script_on_multiple_datasets(
            pretrained_model_path=model_path,
            dump_result_path=eval_result_path,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config_path", type=str, default="configs/mix_teacher_config.yaml")
    p.add_argument("--pretrained_dataset", type=str, default="fgvc-aircraft")
    p.add_argument("--dataset", type=str, default="caltech-101")

    args = p.parse_args()

    main(args)
