import argparse
from itertools import product

from scripts.utils import train_and_eval_script


def main(args):
    params_list_dict = {
        "params.ratio_mix": [10],
        "params.normalize": [False],
    }

    for params in product(*params_list_dict.values()):
        params_dict = dict(zip(params_list_dict.keys(), params))

        train_and_eval_script(
            config_path=args.config_path,
            training_dataset=args.dataset,
            pretrained_dataset=args.pretrained_dataset,
            **params_dict
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config_path", type=str, default="configs/mix_teacher_config.yaml")
    p.add_argument("--pretrained_dataset", type=str, default="fgvc-aircraft")
    p.add_argument("--dataset", type=str, default="dtd")

    args = p.parse_args()

    main(args)
