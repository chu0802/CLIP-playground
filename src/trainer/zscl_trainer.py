import numpy as np
import torch

from src.trainer.base_trainer import BaseKDTrainer
from src.trainer.utils import L2Loss


class ReferenceTrainer(BaseKDTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_criterion = L2Loss(reduce=None, square=False)

    def teacher_model(self, teacher_source):
        return self._teachers[teacher_source]

    def reference_loss(
        self,
        images,
        labels,
        ratio_ref=1,
        label_smoothing=0.0,
        teacher_source="pretrained",
        **_,
    ):
        ref_images, _ = self.get_ref_data(self.ref_loader)

        base_loss, loss_dict = self.base_loss(
            images, labels, label_smoothing=label_smoothing
        )

        with torch.no_grad():
            (teacher_ref_image_embedding, _, _,) = self.teacher_model(
                teacher_source
            )(ref_images, get_features=True)

        student_ref_image_embedding = self.unwrapped_model(self.train_model).encode(
            images=ref_images
        )

        kd_loss = self._get_kd_loss(
            student_ref_image_embedding,
            teacher_ref_image_embedding,
            feature_criterion=self.feature_criterion,
        )

        return (
            base_loss + ratio_ref * kd_loss,
            {
                **loss_dict,
                "kd_loss": kd_loss.item(),
            },
        )


class ZSCLTrainer(BaseKDTrainer):
    @property
    def l2_model(self):
        return self._teachers["l2"]

    @property
    def ref_sentences(self):
        return self.dataloaders["ref_sentences"]

    def l2_loss(self, model, model_ref):
        loss = 0.0
        for param_q, param_k in zip(model.parameters(), model_ref.parameters()):
            loss += torch.nn.functional.mse_loss(
                param_q, param_k.detach(), reduction="sum"
            )

        return loss

    def zscl_loss(self, images, labels, label_smoothing=0.2, l2_ratio=1, **_):
        ref_images, _ = self.get_ref_data(self.ref_loader)

        base_loss, loss_dict = self.base_loss(
            images, labels, label_smoothing=label_smoothing
        )

        with torch.no_grad():
            (
                teacher_ref_image_embedding,
                teacher_ref_text_embedding,
                logit_scale,
            ) = self.pretrained_teacher_model(
                ref_images, self.ref_sentences, get_features=True
            )

        student_ref_image_embedding = self.unwrapped_model(self.train_model).encode(
            images=ref_images
        )

        student_logits = (
            logit_scale * student_ref_image_embedding @ teacher_ref_text_embedding.t()
        )
        teacher_logits = (
            logit_scale * teacher_ref_image_embedding @ teacher_ref_text_embedding.t()
        )

        zscl_image_loss = self._get_kd_loss(student_logits, teacher_logits)

        student_text_logits = student_logits.t()
        teacher_text_logits = teacher_logits.t()

        zscl_text_loss = self._get_kd_loss(student_text_logits, teacher_text_logits)

        loss = base_loss + zscl_image_loss + zscl_text_loss
        loss_dict.update(
            {
                "zscl_image_loss": zscl_image_loss.item(),
                "zscl_text_loss": zscl_text_loss.item(),
            }
        )

        if l2_ratio > 0:
            l2_loss = self.l2_loss(self.train_model, self.l2_model)

            loss += l2_ratio * l2_loss
            loss_dict.update({"l2_loss": l2_loss.item()})

        return loss, loss_dict

    def train(self, *args, **kwargs):
        self.dataloaders["ref_sentences"] = self.unwrapped_model(
            self.train_model
        ).tokenize(self.dataloaders["ref_sentences"])
        super().train(*args, **kwargs)


class PreviousAwareZSCLTrainer(ZSCLTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prev_teacher_model.eval()

    @property
    def prev_teacher_model(self):
        return self._teachers["prev"]

    @property
    def previous_loader(self):
        return self.dataloaders["prev"]

    def previous_aware_zscl_loss(
        self,
        images,
        labels,
        ratio_ref,
        ratio_prev,
        has_noise=False,
        label_smoothing=0.2,
        **_,
    ):
        zscl_loss, loss_dict = self.zscl_loss(
            images,
            labels,
            ratio_ref,
            feature_criterion=self.feature_criterion,
            label_smoothing=label_smoothing,
        )
        previous_images, _ = self.get_ref_data(
            self.previous_loader, has_noise=has_noise
        )

        previous_loss = self.get_kd_loss(
            previous_images,
            teacher_model=self.prev_teacher_model,
            feature_criterion=self.feature_criterion,
        )
        loss_dict.update({"previous_loss": previous_loss.item()})
        return zscl_loss + ratio_prev * previous_loss, loss_dict

    def train(self, *args, **kwargs):
        self.dataloaders["prev"].init()
        super().train(*args, **kwargs)
