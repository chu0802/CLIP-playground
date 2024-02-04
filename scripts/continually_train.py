import argparse

from scripts.utils import DEFAULT_DATASET_SEQ, ContinualTrainer


def main(args):
    continual_trainer = ContinualTrainer(
        config_path=args.config_path,
        training_dataset_seq=args.dataset_seq.split(","),
    )

    continual_trainer.train_and_eval()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config_path", type=str, default="configs/mix_teacher_config.yaml")
    p.add_argument(
        "--dataset_seq",
        type=str,
        default=",".join(DEFAULT_DATASET_SEQ),
        help="the sequence of training datasets, splitted by comma",
    )
    args = p.parse_args()

    main(args)
