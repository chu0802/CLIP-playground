import argparse

from scripts.utils import DEFAULT_DATASET_SEQ, ContinualTrainer


def main(args):
    continual_trainer = ContinualTrainer(
        config_path=args.config_path,
        training_dataset_seq=args.dataset_seq.split(","),
        distributed=args.distributed,
        nnodes=args.nnodes,
        nproc_per_node=args.nproc_per_node,
    )

    continual_trainer.train_and_eval(args.pretrained_dataset)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config_path", type=str, default="configs/mix_teacher_config.yaml")
    p.add_argument(
        "--dataset_seq",
        type=str,
        default=",".join(DEFAULT_DATASET_SEQ),
        help="the sequence of training datasets, splitted by comma",
    )
    p.add_argument("--pretrained_dataset", type=str, default=None)
    p.add_argument("--distributed", action="store_true")
    p.add_argument("--nnodes", type=int, default=1)
    p.add_argument("--nproc_per_node", type=int, default=1)
    args = p.parse_args()

    main(args)
