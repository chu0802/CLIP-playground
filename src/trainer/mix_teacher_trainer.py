import torch
import torch.nn as nn

from src.trainer.base_trainer import BaseKDTrainer
from src.trainer.utils import L2Loss


class MixTeacherKDTrainer(BaseKDTrainer):
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
        base_loss, loss_dict = self.base_loss(
            images, labels, label_smoothing=label_smoothing
        )

        student_logits = self.train_model(ref_images, get_features=True)

        with torch.no_grad():
            pretrained_teacher_logits = self.pretrained_teacher_model(
                ref_images, get_features=True
            )
            prev_teacher_logits = self.prev_teacher_model(ref_images, get_features=True)

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
            **loss_dict,
            "mix_teacher_loss": mix_teacher_loss.item(),
            "num_valid_prev_data": self.num_valid_prev_data,
        }


class SplitTeacherKDTrainer(MixTeacherKDTrainer):
    def split_teacher_loss(
        self,
        images,
        labels,
        ratio_prev=9,
        ratio_pretrained=0.5,
        threshold=0.2,
        scale=6,
        label_smoothing=0.0,
    ):
        ref_images, _ = self.get_ref_data(self.ref_loader)
        base_loss, loss_dict = self.base_loss(
            images, labels, label_smoothing=label_smoothing
        )

        student_logits = self.train_model(ref_images, get_features=True)

        with torch.no_grad():
            pretrained_teacher_logits = self.pretrained_teacher_model(
                ref_images, get_features=True
            )
            prev_teacher_logits = self.prev_teacher_model(ref_images, get_features=True)

        pre_scores = torch.norm(pretrained_teacher_logits - prev_teacher_logits, dim=-1)

        self.num_valid_prev_data += (pre_scores > threshold).float().sum().item()

        scaled_scores = scale * (pre_scores - threshold)

        scores = nn.functional.sigmoid(scaled_scores).reshape(-1, 1)

        pretrained_kd_loss = self._get_kd_loss(
            student_logits,
            pretrained_teacher_logits,
            feature_criterion=self.feature_criterion,
        )
        prev_kd_loss = self._get_kd_loss(
            student_logits,
            prev_teacher_logits,
            feature_criterion=self.feature_criterion,
        )

        prev_kd_loss = (scores * prev_kd_loss).mean()

        pretrained_kd_loss = ((1 - scores) * pretrained_kd_loss).mean()

        return (
            base_loss
            + ratio_prev * prev_kd_loss
            + ratio_pretrained * pretrained_kd_loss,
            {
                **loss_dict,
                "prev_kd_loss": prev_kd_loss.item(),
                "pretrained_kd_loss": pretrained_kd_loss.item(),
                "num_valid_prev_data": self.num_valid_prev_data,
            },
        )


class SplitTeacherPureClipKDTrainer(MixTeacherKDTrainer):
    def split_teacher_pure_clip_loss(
        self,
        images,
        labels,
        ratio_prev=9,
        ratio_pretrained=0.5,
        threshold=0.2,
        scale=6,
        label_smoothing=0.0,
    ):
        ref_images, _ = self.get_ref_data(self.ref_loader)
        base_loss, loss_dict = self.base_loss(
            images, labels, label_smoothing=label_smoothing
        )

        student_ref_image_embedding = self.unwrapped_model(self.train_model).encode(
            images=ref_images
        )

        with torch.no_grad():
            (
                pretrained_teacher_ref_image_embedding,
                _,
                _,
            ) = self.pretrained_teacher_model(ref_images, get_features=True)

            (
                prev_teacher_ref_image_embedding,
                _,
                _,
            ) = self.prev_teacher_model(ref_images, get_features=True)

        pre_scores = torch.norm(
            pretrained_teacher_ref_image_embedding - prev_teacher_ref_image_embedding,
            dim=-1,
        )

        self.num_valid_prev_data += (pre_scores > threshold).float().sum().item()

        scaled_scores = scale * (pre_scores - threshold)

        scores = nn.functional.sigmoid(scaled_scores).reshape(-1, 1)

        pretrained_kd_loss = self._get_kd_loss(
            student_ref_image_embedding,
            pretrained_teacher_ref_image_embedding,
            feature_criterion=self.feature_criterion,
        )
        prev_kd_loss = self._get_kd_loss(
            student_ref_image_embedding,
            prev_teacher_ref_image_embedding,
            feature_criterion=self.feature_criterion,
        )

        prev_kd_loss = (scores * prev_kd_loss).mean()

        pretrained_kd_loss = ((1 - scores) * pretrained_kd_loss).mean()

        return (
            base_loss
            + ratio_prev * prev_kd_loss
            + ratio_pretrained * pretrained_kd_loss,
            {
                **loss_dict,
                "prev_kd_loss": prev_kd_loss.item(),
                "pretrained_kd_loss": pretrained_kd_loss.item(),
                "num_valid_prev_data": self.num_valid_prev_data,
            },
        )


class SplitTeacherPureClipFixedScoresTrainer(MixTeacherKDTrainer):
    def split_teacher_pure_clip_fixed_scores_loss(
        self,
        images,
        labels,
        ratio_prev=9,
        ratio_pretrained=0.5,
        label_smoothing=0.0,
        scores=0.5,
    ):
        ref_images, _ = self.get_ref_data(self.ref_loader)
        base_loss, loss_dict = self.base_loss(
            images, labels, label_smoothing=label_smoothing
        )

        student_ref_image_embedding = self.unwrapped_model(self.train_model).encode(
            images=ref_images
        )

        with torch.no_grad():
            (
                pretrained_teacher_ref_image_embedding,
                _,
                _,
            ) = self.pretrained_teacher_model(ref_images, get_features=True)

            (
                prev_teacher_ref_image_embedding,
                _,
                _,
            ) = self.prev_teacher_model(ref_images, get_features=True)

        pretrained_kd_loss = self._get_kd_loss(
            student_ref_image_embedding,
            pretrained_teacher_ref_image_embedding,
            feature_criterion=self.feature_criterion,
        )
        prev_kd_loss = self._get_kd_loss(
            student_ref_image_embedding,
            prev_teacher_ref_image_embedding,
            feature_criterion=self.feature_criterion,
        )

        prev_kd_loss = (scores * prev_kd_loss).mean()

        pretrained_kd_loss = ((1 - scores) * pretrained_kd_loss).mean()

        return (
            base_loss
            + ratio_prev * prev_kd_loss
            + ratio_pretrained * pretrained_kd_loss,
            {
                **loss_dict,
                "prev_kd_loss": prev_kd_loss.item(),
                "pretrained_kd_loss": pretrained_kd_loss.item(),
                "num_valid_prev_data": self.num_valid_prev_data,
            },
        )


class PretrainedTeacherTrainer(MixTeacherKDTrainer):
    def pretrained_teacher_loss(
        self,
        images,
        labels,
        label_smoothing=0.0,
    ):
        ref_images, _ = self.get_ref_data(self.ref_loader)
        base_loss, loss_dict = self.base_loss(
            images, labels, label_smoothing=label_smoothing
        )

        student_ref_image_embedding = self.unwrapped_model(self.train_model).encode(
            images=ref_images
        )

        with torch.no_grad():
            (
                pretrained_teacher_ref_image_embedding,
                _,
                _,
            ) = self.pretrained_teacher_model(ref_images, get_features=True)

        pretrained_kd_loss = self._get_kd_loss(
            student_ref_image_embedding,
            pretrained_teacher_ref_image_embedding,
            feature_criterion=self.feature_criterion,
        ).mean()

        return (
            base_loss + pretrained_kd_loss,
            {
                **loss_dict,
                "pretrained_kd_loss": pretrained_kd_loss.item(),
                "num_valid_prev_data": self.num_valid_prev_data,
            },
        )


class PrevTeacherTrainer(MixTeacherKDTrainer):
    def prev_teacher_loss(
        self,
        images,
        labels,
        label_smoothing=0.0,
    ):
        ref_images, _ = self.get_ref_data(self.ref_loader)
        base_loss, loss_dict = self.base_loss(
            images, labels, label_smoothing=label_smoothing
        )

        student_ref_image_embedding = self.unwrapped_model(self.train_model).encode(
            images=ref_images
        )

        with torch.no_grad():
            (
                prev_teacher_ref_image_embedding,
                _,
                _,
            ) = self.prev_teacher_model(ref_images, get_features=True)

        prev_kd_loss = self._get_kd_loss(
            student_ref_image_embedding,
            prev_teacher_ref_image_embedding,
            feature_criterion=self.feature_criterion,
        ).mean()

        return (
            base_loss + prev_kd_loss,
            {
                **loss_dict,
                "prev_kd_loss": prev_kd_loss.item(),
                "num_valid_prev_data": self.num_valid_prev_data,
            },
        )
