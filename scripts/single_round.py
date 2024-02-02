import argparse

from scripts import evaluation_script_on_multiple_datasets, training_script
from scripts.utils import get_model_path, get_output_dataset_dir


def main(args):
    pretrained_path = get_model_path(args.pretrained_dataset)

    training_script(
        config_path=args.config_path,
        training_script="kd_train.py",
        dataset=args.dataset,
        pretrained_model_path=pretrained_path,
    )

    # evaluation
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
