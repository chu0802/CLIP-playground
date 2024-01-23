from itertools import product
from pathlib import Path

from scripts import evaluation_script_on_multiple_datasets, training_script

OUTPUTS_ROOT = Path("outputs/ViT-B-16")
TRAIN_DATASET = "caltech-101"
TEST_DATASET = "fgvc-aircraft"
PRETRAINED_PATH = "outputs/ViT-B-16/fgvc-aircraft/latest/checkpoint_1.pth"

if __name__ == "__main__":
    ratio_prev_list = [9]
    threshold_list = [0.2]
    scale_list = [6]

    for ratio_prev, threshold, scale in product(
        ratio_prev_list, threshold_list, scale_list
    ):
        training_script(
            config_path="configs/mix_teacher_config.yaml",
            training_script="kd_train.py",
            dataset=TRAIN_DATASET,
            pretrained_model_path=PRETRAINED_PATH,
            sample_num=10,
            max_epoch=1,
            ratio_prev=ratio_prev,
            threshold=threshold,
            scale=scale,
        )

        model_path = OUTPUTS_ROOT / TRAIN_DATASET / "latest" / "checkpoint_1.pth"
        eval_result_path = OUTPUTS_ROOT / TRAIN_DATASET / "latest" / "eval_results.json"

        eval_results = evaluation_script_on_multiple_datasets(
            datasets=[TEST_DATASET],
            pretrained_model_path=model_path,
            sample_num=10,
            dump_result_path=eval_result_path,
        )
