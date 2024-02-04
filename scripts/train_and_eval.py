import argparse

from scripts.utils import train_and_eval_script


def main(args):
    train_and_eval_script(
        config_path=args.config_path,
        training_dataset=args.dataset,
        pretrained_dataset=args.pretrained_dataset,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config_path", type=str, default="configs/mix_teacher_config.yaml")
    p.add_argument("--pretrained_dataset", type=str, default="fgvc-aircraft")
    p.add_argument("--dataset", type=str, default="caltech-101")

    args = p.parse_args()

    main(args)
