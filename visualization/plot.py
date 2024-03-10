import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"


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


plot_figure(
    {
        "Continual-FT": [
            54.1254,
            44.6145,
            22.5623,
            27.3627,
            21.6922,
            22.6823,
            21.5722,
            23.7924,
        ],
        "LwF": [54.3654, 41.5242, 33.1833, 32.3132, 27.3927, 25.8926, 21.1821, 17.5218],
        "iCaRL": [
            54.0654,
            51.3051,
            40.4440,
            36.2136,
            35.2535,
            35.8536,
            28.4728,
            29.3429,
        ],
        "ZSCL": [
            53.2253,
            49.9850,
            44.0444,
            39.5440,
            36.9337,
            35.2235,
            32.9133,
            34.2634,
        ],
        "Ours": [
            52.1152,
            52.1152,
            51.9052,
            52.3252,
            51.2451,
            50.9751,
            47.3747,
            46.4146,
        ],
    },
    zero_shot=23.91,
    title="Acc. of the 1st task in $\mathcal{S}^1$ (Aircraft)",
    legend="lower left",
    save_path="aircraft_forgetting.pdf",
)

plot_figure(
    data_dict={
        "base": [62.07, 56.67, 44.91, 50.73, 55.09, 50.75, 54.59, 87.84],
        "LwF": [63.47, 61.56, 57.6, 57.65, 59.03, 58.68, 60.08, 86.1],
        "iCaRL": [64.16, 61.41, 54.53, 56.78, 59.64, 56.81, 59.16, 86.52],
        "ZSCL": [63.10, 62.33, 59.5, 60.69, 62.23, 61.88, 63.52, 88.21],
        "Ours": [62.94, 62.70, 62.33, 62.28, 62.44, 62.54, 62.31, 88.08],
    },
    zero_shot=64.26,
    title="Acc. of the 8th task in $\mathcal{S}^1$ (UCF101)",
    legend="upper left",
    save_path="ucf101_degradation.pdf",
)

plot_figure(
    {
        "Continual-FT": [
            94.1401,
            93.5677,
            88.689,
            90.3243,
            90.8149,
            78.8771,
            88.5527,
            82.5566,
        ],
        "LwF": [94.0856, 93.5677, 92.0959, 91.3055, 90.7604, 88.2529, 87.7351, 88.8798],
        "iCaRL": [
            94.3309,
            93.3497,
            89.7792,
            91.251,
            91.3873,
            84.5735,
            89.2614,
            86.5086,
        ],
        "ZSCL": [
            95.1758,
            94.6579,
            94.0583,
            94.2491,
            93.3769,
            92.2867,
            91.6871,
            90.7332,
        ],
        "Ours": [
            95.5301,
            95.2303,
            95.4484,
            95.5574,
            95.3666,
            95.3938,
            95.2303,
            94.6579,
        ],
    },
    zero_shot=87.27,
    title="Acc. of the 1st task in $\mathcal{S}^3$ (Pets)",
    legend="lower left",
    save_path="pets_forgetting.pdf",
)

plot_figure(
    {
        "Continual-FT": [
            70.7888,
            77.8152,
            72.0396,
            72.264,
            70.5347,
            54.2838,
            62.7393,
            89.4092,
        ],
        "LwF": [75.6766, 77.7327, 76.2838, 76.0165, 73.5908, 66.2442, 66.4917, 87.3234],
        "iCaRL": [
            77.0132,
            74.868,
            75.1353,
            71.9175,
            73.4257,
            66.4323,
            67.0165,
            88.3465,
        ],
        "ZSCL": [
            82.0363,
            81.8581,
            83.3498,
            83.4455,
            83.5347,
            82.6469,
            82.6238,
            89.2376,
        ],
        "Ours": [83.835, 83.4785, 83.4158, 83.0462, 82.4818, 82.3135, 82.099, 89.9637],
    },
    zero_shot=84.00,
    title="Acc. of the 8th task in $\mathcal{S}^3$ (Food)",
    legend="lower left",
    save_path="food_degradation.pdf",
)
