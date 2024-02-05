import argparse

from scripts.utils import train_and_eval_script


def main(args):
    train_and_eval_script(
        config_path=args.config_path,
        training_dataset=args.dataset,
        pretrained_dataset=args.pretrained_dataset,
        max_epoch=args.train_epoch,
        max_iterations=args.max_iterations,
        eval_epoch=args.eval_epoch,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config_path", type=str, default="configs/mix_teacher_config.yaml")
    p.add_argument("--pretrained_dataset", type=str, default="fgvc-aircraft")
    p.add_argument("--dataset", type=str, default="caltech-101")
    p.add_argument("--train_epoch", type=int, default=10)
    p.add_argument("--max_iterations", type=int, default=1000)
    p.add_argument(
        "--eval_epoch",
        type=int,
        default=10,
        help="determine to use the model saved in which epoches to evaluate",
    )

    args = p.parse_args()

    main(args)
