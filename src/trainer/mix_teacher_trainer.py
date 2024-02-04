import torch
import torch.nn as nn

from src.trainer.utils import L2Loss
from src.trainer.zscl_trainer import ZSCLTrainer


class MixTeacherKDTrainer(ZSCLTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prev_teacher_model.eval()
        self.feature_criterion = L2Loss(reduce=None, square=False)
        self.num_valid_prev_data = 0

    @property
    def prev_teacher_model(self):
        return self._teachers["prev"]

    def scoring_function(
        self, pretrained_teacher_logits, prev_teacher_logits, threshold=0.2, scale=6
    ):
        pre_scores = torch.norm(pretrained_teacher_logits - prev_teacher_logits, dim=-1)
        self.num_valid_prev_data += (pre_scores > threshold).float().sum().item()

        scaled_scores = scale * (pre_scores - threshold)

        return nn.functional.sigmoid(scaled_scores).reshape(-1, 1)

    def get_mix_teacher_feature(
        self,
        pretrained_teacher_logits,
        prev_teacher_logits,
        threshold=0.2,
        scale=6,
        normalize=False,
    ):
        scores = self.scoring_function(
            pretrained_teacher_logits, prev_teacher_logits, threshold, scale
        )
        mix_teacher_feature = (
            1 - scores
        ) * pretrained_teacher_logits + scores * prev_teacher_logits

        if normalize:
            mix_teacher_feature = torch.nn.functional.normalize(
                mix_teacher_feature, p=2, dim=1
            )

        return mix_teacher_feature

    def mix_teacher_loss(
        self,
        images,
        labels,
        threshold=0.2,
        scale=6,
        ratio_mix=2,
        normalize=False,
        label_smoothing=0.2,
    ):
        ref_images, _ = self.get_ref_data(self.ref_loader)
        base_loss = self.base_loss(images, labels, label_smoothing=label_smoothing)

        student_logits = self.train_model.get_features(ref_images)

        with torch.no_grad():
            pretrained_teacher_logits = self.pretrained_teacher_model.get_features(
                ref_images
            )
            prev_teacher_logits = self.prev_teacher_model.get_features(ref_images)

        mix_teacher_feature = self.get_mix_teacher_feature(
            pretrained_teacher_logits,
            prev_teacher_logits,
            threshold,
            scale,
            normalize=normalize,
        )

        mix_teacher_loss = self._get_kd_loss(
            student_logits,
            mix_teacher_feature,
            feature_criterion=self.feature_criterion,
        ).mean()

        return base_loss + ratio_mix * mix_teacher_loss, {
            "base_loss": base_loss.item(),
            "mix_teacher_loss": mix_teacher_loss.item(),
            "num_valid_prev_data": self.num_valid_prev_data,
        }
