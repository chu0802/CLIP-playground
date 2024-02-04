import numpy as np
import torch

from src.trainer.base_trainer import BaseKDTrainer
from src.trainer.utils import L2Loss


class ZSCLTrainer(BaseKDTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_criterion = L2Loss(reduce="mean")

    @property
    def ref_loader(self):
        return self.dataloaders["ref"]

    def get_ref_data(self, loader, has_noise=False):
        try:
            ref_data = next(loader)
        except StopIteration:
            loader.init()
            ref_data = next(loader)
        data, index = ref_data[0], ref_data[-1]
        if has_noise:
            data += ref_data[1]

        return data, index

    def zscl_loss(self, images, labels, ratio, label_smoothing=0.2, **_):
        ref_images, _ = self.get_ref_data(self.ref_loader)
        return self.general_kd_loss(
            images,
            labels,
            ref_images,
            ratio,
            feature_criterion=self.feature_criterion,
            label_smoothing=label_smoothing,
        )

    def train(self, *args, **kwargs):
        self.dataloaders["ref"].init()
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
        mixup=False,
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

        if mixup:
            permute_images = previous_images[
                torch.randperm(previous_images.shape[0]).to(previous_images.device)
            ]
            lamda = np.random.beta(1.0, 1.0)
            previous_images = lamda * previous_images + (1 - lamda) * permute_images

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
