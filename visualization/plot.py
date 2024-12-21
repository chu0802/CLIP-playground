import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from scripts.metrics import DEFAULT_ZERO_SHOT_PERFORMANCE
from scripts.utils import DEFAULT_DATASET_SEQ

plt.rcParams["font.family"] = "Times New Roman"

METHOD_MAP = {
    "Continual-FT": "base",
    "LwF": "lwf",
    "iCaRL": "icarl",
    "ZSCL": "zscl",
}

VISUALIZED_DATASET_NAME_MAP = {
    "fgvc-aircraft": "Aircraft",
    "dtd": "DTD",
    "eurosat": "EuroSAT",
    "flowers-102": "Flowers",
    "food-101": "Food",
    "oxford-pets": "Pets",
    "stanford-cars": "Cars",
    "ucf-101": "UCF101",
}


def parse_results(method="split_teacher_pure_clip", is_mdcil=False):
    config_name = f"{method}_config"

    res_list = []
    for order in range(8):
        res_path = (
            Path("/work/chu980802/mix-teacher")
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


def plot_figure(
    data_dict, zero_shot, title, legend=None, save_path="comparison_plot.pdf"
):
    plt.figure(figsize=(10, 10))
    markers = ["o", "s", "D", "^", "P"]  # Different markers for each series
    # colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    colors = ["#92CD00", "#A3D1F2", "#F4B6C2", "#FED1BD", "#957DAD", "#88D8C0"]

    plt.scatter(
        0, zero_shot, marker="*", s=800, color="#88D8C0", label="Zero-shot", zorder=3
    )

    for (label, data), marker, color in zip(data_dict.items(), markers, colors):
        plt.plot(
            range(len(data) + 1),
            [zero_shot] + data,
            label=label,
            marker=marker,
            lw=3,
            markersize=10,
            color=color,
        )
    plt.title(title, fontsize=40)
    plt.xlabel("Task sequence", fontsize=40)
    plt.ylabel("Accuracy", fontsize=40)
    plt.tick_params(labelsize=30)
    plt.xticks(range(9), range(0, 9))  # Example x-axis labels
    if legend is not None:
        plt.legend(fontsize=26, loc=legend)
    plt.grid(linestyle=":")
    plt.tight_layout()
    # Save the figure as a PDF
    plt.savefig(save_path)


res_list_dict = {k: parse_results(v) for k, v in METHOD_MAP.items()}

res_list = [
    {method: res[order] for method, res in res_list_dict.items()} for order in range(8)
]

# Catastrophic forgetting
for order in range(8):
    dataset_name = {res.index[0] for res in res_list[order].values()}
    assert len(dataset_name) == 1
    dataset_name = dataset_name.pop()
    display_order = DEFAULT_DATASET_SEQ.index(dataset_name)
    legend = "lower left" if order == 0 else None
    plot_figure(
        {
            method: res.loc[:, dataset_name].values.tolist()
            for method, res in res_list[order].items()
        },
        zero_shot=DEFAULT_ZERO_SHOT_PERFORMANCE[dataset_name],
        title="Acc. of the 1st task in $\mathcal{S}^<ORDER>$ (<DATASET>)".replace(
            "<ORDER>", str(display_order + 1)
        ).replace("<DATASET>", VISUALIZED_DATASET_NAME_MAP[dataset_name]),
        legend=legend,
        save_path=f"visualization/{dataset_name}_forgetting_order_{display_order+1}.pdf",
    )

# Zero-shot degradation
for order in range(8):
    dataset_name = {res.index[-1] for res in res_list[order].values()}
    assert len(dataset_name) == 1
    dataset_name = dataset_name.pop()
    display_order = (
        DEFAULT_DATASET_SEQ.index(dataset_name)
        + 1
        - 8 * (DEFAULT_DATASET_SEQ.index(dataset_name) == 7)
    )
    legend = "upper left" if order == 0 else None
    plot_figure(
        {
            method: res.loc[:, dataset_name].values.tolist()
            for method, res in res_list[order].items()
        },
        zero_shot=DEFAULT_ZERO_SHOT_PERFORMANCE[dataset_name],
        title="Acc. of the 8th task in $\mathcal{S}^<ORDER>$ (<DATASET>)".replace(
            "<ORDER>", str(display_order + 1)
        ).replace("<DATASET>", VISUALIZED_DATASET_NAME_MAP[dataset_name]),
        legend=legend,
        save_path=f"visualization/{dataset_name}_degradation_order_{display_order+1}.pdf",
    )

# plot_figure(
#     {
#         "Continual-FT": [
#             54.1254,
#             44.6145,
#             22.5623,
#             27.3627,
#             21.6922,
#             22.6823,
#             21.5722,
#             23.7924,
#         ],
#         "LwF": [54.3654, 41.5242, 33.1833, 32.3132, 27.3927, 25.8926, 21.1821, 17.5218],
#         "iCaRL": [
#             54.0654,
#             51.3051,
#             40.4440,
#             36.2136,
#             35.2535,
#             35.8536,
#             28.4728,
#             29.3429,
#         ],
#         "ZSCL": [
#             53.2253,
#             49.9850,
#             44.0444,
#             39.5440,
#             36.9337,
#             35.2235,
#             32.9133,
#             34.2634,
#         ],
#         "Ours": [
#             52.1152,
#             52.1152,
#             51.9052,
#             52.3252,
#             51.2451,
#             50.9751,
#             47.3747,
#             46.4146,
#         ],
#     },
#     zero_shot=23.91,
#     title="Acc. of the 1st task in $\mathcal{S}^1$ (Aircraft)",
#     legend="lower left",
#     save_path="aircraft_forgetting.pdf",
# )

# plot_figure(
#     data_dict={
#         "base": [62.07, 56.67, 44.91, 50.73, 55.09, 50.75, 54.59, 87.84],
#         "LwF": [63.47, 61.56, 57.6, 57.65, 59.03, 58.68, 60.08, 86.1],
#         "iCaRL": [64.16, 61.41, 54.53, 56.78, 59.64, 56.81, 59.16, 86.52],
#         "ZSCL": [63.10, 62.33, 59.5, 60.69, 62.23, 61.88, 63.52, 88.21],
#         "Ours": [62.94, 62.70, 62.33, 62.28, 62.44, 62.54, 62.31, 88.08],
#     },
#     zero_shot=64.26,
#     title="Acc. of the 8th task in $\mathcal{S}^1$ (UCF101)",
#     legend="upper left",
#     save_path="ucf101_degradation.pdf",
# )

# plot_figure(
#     {
#         "Continual-FT": [
#             94.1401,
#             93.5677,
#             88.689,
#             90.3243,
#             90.8149,
#             78.8771,
#             88.5527,
#             82.5566,
#         ],
#         "LwF": [94.0856, 93.5677, 92.0959, 91.3055, 90.7604, 88.2529, 87.7351, 88.8798],
#         "iCaRL": [
#             94.3309,
#             93.3497,
#             89.7792,
#             91.251,
#             91.3873,
#             84.5735,
#             89.2614,
#             86.5086,
#         ],
#         "ZSCL": [
#             95.1758,
#             94.6579,
#             94.0583,
#             94.2491,
#             93.3769,
#             92.2867,
#             91.6871,
#             90.7332,
#         ],
#         "Ours": [
#             95.5301,
#             95.2303,
#             95.4484,
#             95.5574,
#             95.3666,
#             95.3938,
#             95.2303,
#             94.6579,
#         ],
#     },
#     zero_shot=87.27,
#     title="Acc. of the 1st task in $\mathcal{S}^3$ (Pets)",
#     legend="lower left",
#     save_path="pets_forgetting.pdf",
# )

# plot_figure(
#     {
#         "Continual-FT": [
#             70.7888,
#             77.8152,
#             72.0396,
#             72.264,
#             70.5347,
#             54.2838,
#             62.7393,
#             89.4092,
#         ],
#         "LwF": [75.6766, 77.7327, 76.2838, 76.0165, 73.5908, 66.2442, 66.4917, 87.3234],
#         "iCaRL": [
#             77.0132,
#             74.868,
#             75.1353,
#             71.9175,
#             73.4257,
#             66.4323,
#             67.0165,
#             88.3465,
#         ],
#         "ZSCL": [
#             82.0363,
#             81.8581,
#             83.3498,
#             83.4455,
#             83.5347,
#             82.6469,
#             82.6238,
#             89.2376,
#         ],
#         "Ours": [83.835, 83.4785, 83.4158, 83.0462, 82.4818, 82.3135, 82.099, 89.9637],
#     },
#     zero_shot=84.00,
#     title="Acc. of the 8th task in $\mathcal{S}^3$ (Food)",
#     legend="lower left",
#     save_path="food_degradation.pdf",
# )
