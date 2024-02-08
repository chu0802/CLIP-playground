import argparse

from scripts.utils import DEFAULT_DATASET_SEQ, ContinualTrainer
from collections import deque
from copy import deepcopy

def main(args):
    if args.dataset_seq is None:
        args.dataset_seq = deque(deepcopy(DEFAULT_DATASET_SEQ))
        args.dataset_seq.rotate(args.order)
    else:
        args.dataset_seq = args.dataset_seq.split(",")

    continual_trainer = ContinualTrainer(
        config_path=args.config_path,
        training_dataset_seq=args.dataset_seq,
        order=args.order,
        distributed=args.distributed,
        nnodes=args.nnodes,
        nproc_per_node=args.nproc_per_node,
    )

    continual_trainer.train_and_eval(args.pretrained_dataset)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config_path", type=str, default="configs/split_teacher_config.yaml")
    p.add_argument(
        "--dataset_seq",
        type=str,
        default=None,
        help="the sequence of training datasets, splitted by comma",
    )
    p.add_argument("--pretrained_dataset", type=str, default=None)
    p.add_argument("--distributed", action="store_true")
    p.add_argument("--nnodes", type=int, default=1)
    p.add_argument("--nproc_per_node", type=int, default=1)
    p.add_argument("--order", type=int, default=0)
    args = p.parse_args()

    main(args)
