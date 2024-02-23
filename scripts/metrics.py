import argparse
import json

import pandas as pd

from scripts.utils import DEFAULT_DATASET_SEQ, DEFAULT_STORAGE_ROOT

DEFAULT_ZERO_SHOT_PERFORMANCE = {
    "fgvc-aircraft": 0.2391,
    "caltech-101": 0.9221,
    "dtd": 0.4439,
    "eurosat": 0.4222,
    "flowers-102": 0.6740,
    "food-101": 0.84,
    "oxford-pets": 0.8727,
    "stanford-cars": 0.6551,
    "ucf-101": 0.6426,
}


def metric_to_dataframe(metric, index_name, columns=DEFAULT_DATASET_SEQ):
    data_frame = pd.DataFrame(metric, index=[index_name]).loc[:, columns]
    return data_frame.round(2)


def max_catastrophic_forgetting(res_list):
    metric = {
        res.index[0]: 100
        * (res.loc[:, res.index[0]].min() - res.loc[:, res.index[0]].max())
        # res.index[0]: 100 * res.loc[:, res.index[0]].min()
        for res in res_list
    }
    return metric_to_dataframe(metric, "catastrophic forgetting")


def max_zero_shot_degradation(res_list):
    metric = {
        res.index[-1]: 100
        * (
            res.loc[:, res.index[-1]].min()
            - DEFAULT_ZERO_SHOT_PERFORMANCE[res.index[-1]]
        )
        # res.index[-1]: 100 * res.loc[:, res.index[-1]].min()
        for res in res_list
    }
    return metric_to_dataframe(metric, "zero-shot degradation")


def avg_final_performance(res_list):
    metric = 100 * pd.concat(
        [res.iloc[-1][DEFAULT_DATASET_SEQ] for res in res_list], axis=1
    ).mean(axis=1)
    return metric.to_frame("avg. final performance").T


def parse_results(method="split_teacher_pure_clip"):
    if "mdcil" in method:
        config_name = f'{"_".join(method.split("_")[:-1])}_config'
    else:
        config_name = f"{method}_config"
    res_list = []
    for order in range(8):
        res_path = (
            DEFAULT_STORAGE_ROOT
            / method
            / "outputs"
            / f"order_{order}"
            / config_name
            / "final_results.json"
        )
        with res_path.open("r") as f:
            res = json.load(f)
        res_list.append(pd.DataFrame(res).T)
    return res_list


def main(args):
    res_list = parse_results(method=args.method)

    if args.order == "overall":
        forget = max_catastrophic_forgetting(res_list)
        degradation = max_zero_shot_degradation(res_list)
        avg = avg_final_performance(res_list)

        print(pd.concat([forget, degradation, avg], axis=0).round(2))
    else:
        print(
            (100 * res_list[int(args.order)])
            .round(2)
            .loc[:, res_list[int(args.order)].index]
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="split_teacher_pure_clip")
    p.add_argument(
        "--order",
        type=str,
        default="overall",
        choices=[str(i) for i in range(8)] + ["overall"],
    )
    args = p.parse_args()

    main(args)
