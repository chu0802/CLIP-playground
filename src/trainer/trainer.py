import datetime
import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import wandb
from src.datasets import DATASET_MAPPING
from src.datasets.base import NoisyImageListDataset
from src.datasets.utils import build_iter_dataloader, get_dataloader, load_transform
from src.trainer.optimizer import get_optimizer
from src.trainer.scheduler import CosineLRScheduler
from src.utils import AccuracyMeter, dump_config, inference_feature_distance


class CosineSimilarityLoss(nn.CosineEmbeddingLoss):
    def forward(self, x, y):
        return super().forward(x, y, torch.ones(x.shape[0]).to(x.device))


class L2Loss(nn.Module):
    def __init__(self, reduce=None, square=False):
        super().__init__()
        self.reduce = reduce
        self.square = square

    def forward(self, x, y):
        loss = torch.pow(torch.norm(x - y, dim=-1), 2)
        if self.square:
            loss = loss**2
        if self.reduce == "mean":
            return loss.mean()
        return loss


class Trainer:
    def __init__(self, model, dataloaders, config, dump_result=True):
        self.model = model
        self.dataloaders = dataloaders
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        self.dump_result = dump_result

        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        if self.dump_result or self.training_mode:
            self.output_dir = (
                Path(self.config.task.output_dir)
                / self.config.model.vit_base
                / self.config.data.name
                / timestamp
            )
            self.output_dir.mkdir(parents=True, exist_ok=True)
            dump_config(self.config, self.output_dir / "config.json")

        self.local_log = defaultdict(dict)

        if self.training_mode:
            self.optimizer = get_optimizer(self.model, self.config.task)
            self.lr_scheduler = CosineLRScheduler(
                self.optimizer, self.config.task, self.num_total_train_steps
            )
            self.lastest_dir = self.output_dir.parent / "latest"

            if self.lastest_dir.exists():
                # unlink it since it's a symbolic link
                self.lastest_dir.unlink()

            self.lastest_dir.symlink_to(self.output_dir.name)

    def save(self, epoch):
        # TODO: check if freeze classification head or not
        visual_state_dict = self.model.clip_base.model.visual.state_dict()

        save_obj = {"model": visual_state_dict}

        save_path = self.output_dir / f"checkpoint_{epoch}.pth"

        print(f"Saving checkpoint at epoch {epoch} to {save_path}.")
        torch.save(save_obj, save_path)

    @property
    def method_config(self):
        return self.config.method

    @property
    def training_mode(self):
        return self.config.mode == "train"

    @property
    def num_total_train_steps(self):
        return self.max_epoch * len(self.train_loader)

    @property
    def max_epoch(self):
        return self.config.task.max_epoch

    @property
    def lr(self):
        return self.lr_scheduler.current_lr

    @property
    def log_interval(self):
        return self.config.task.log_interval

    @property
    def train_loader(self):
        return self.dataloaders.get("train", None)

    @property
    def val_loader(self):
        return self.dataloaders.get("val", None)

    @property
    def test_loader(self):
        return self.dataloaders.get("test", None)

    def get_current_training_step(self, epoch, local_step):
        return len(self.train_loader) * (epoch - 1) + local_step

    def logging(self, local_desc=None, use_wandb=True, **message_dict):
        if use_wandb:
            wandb.log(message_dict)
        if local_desc is not None:
            self.local_log[local_desc].update(message_dict)

    def dump_results(self, filename="results.json", print_result=False):
        if self.dump_result:
            with open(self.output_dir / filename, "w") as f:
                json.dump(self.local_log, f, indent=4)

        if print_result:
            print(json.dumps(self.local_log))

    def base_loss(self, images, labels, **_):
        outputs = self.model(images)
        return self.criterion(outputs, labels)

    def train_step(self, images, labels):
        # need to step lr_scheduler first since in this repo I didn't explictly set a learning rate in the optimizer.
        self.lr_scheduler.step()

        loss_fn = getattr(self, f"{self.method_config.name}_loss")
        loss, loss_dict = loss_fn(images, labels, **self.method_config.params)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        loss_dict.update({"total_loss": loss.item()})
        return loss_dict

    def evaluate(self, dataloader=None):
        if dataloader is None:
            dataloader = self.test_loader

        self.model.eval()

        scores = AccuracyMeter()

        dataloader.init()
        with torch.no_grad(), tqdm(total=len(dataloader)) as pbar:
            for images, labels, _ in dataloader:
                preds = self.model(images).argmax(dim=1)
                scores += preds == labels
                pbar.set_postfix_str(f"acc: {100 * scores.acc():.2f}%")
                pbar.update(1)

        return scores.acc()

    def train(self, set_validation=False):
        # test zero-shot validation performance
        if self.val_loader and set_validation:
            self.logging(val_acc=self.evaluate(self.val_loader))

        with tqdm(total=self.num_total_train_steps) as pbar:
            for epoch in range(1, self.max_epoch + 1):
                pbar.set_description(f"Epoch {epoch}/{self.max_epoch}: ")

                self.model.train()
                self.train_loader.init()

                for i, (images, labels, _) in enumerate(self.train_loader):
                    loss_dict = self.train_step(images, labels)

                    pbar.set_postfix_str(
                        f"lr: {self.lr:.2e}, loss: {loss_dict['total_loss']:.2e}"
                    )
                    pbar.update(1)

                    if i % self.log_interval == 0:
                        self.logging(lr=self.lr, **loss_dict)

                if self.val_loader and set_validation:
                    self.logging(val_acc=self.evaluate(self.val_loader))

                self.save(epoch)


class KDTrainer(Trainer):
    def __init__(self, model, dataloaders, config, teachers):
        super().__init__(model, dataloaders, config)
        self._teachers = teachers
        self.pretrained_teacher_model.eval()
        self.kl_criterion = nn.KLDivLoss()

    @property
    def pretrained_teacher_model(self):
        return self._teachers["pretrained"]

    def _get_kd_loss(self, student_logits, teacher_logits, feature_criterion=None):
        if feature_criterion:
            return feature_criterion(student_logits, teacher_logits)

        soft_labels = nn.functional.softmax(teacher_logits, dim=-1)
        soft_preds = nn.functional.log_softmax(student_logits, dim=-1)

        return -(soft_labels * soft_preds).sum() / soft_preds.shape[0]

    def get_kd_loss(
        self, ref_data, teacher_model=None, student_logits=None, feature_criterion=None
    ):
        if student_logits is None:
            student_logits = (
                self.model.get_features(ref_data)
                if feature_criterion
                else self.model(ref_data)
            )

        with torch.no_grad():
            if teacher_model is None:
                teacher_model = self.pretrained_teacher_model
            teacher_logits = (
                teacher_model.get_features(ref_data)
                if feature_criterion
                else teacher_model(ref_data)
            )

        return self._get_kd_loss(
            student_logits, teacher_logits, feature_criterion=feature_criterion
        )

    def general_kd_loss(self, images, labels, ref_data, ratio, feature_criterion=None):
        base_loss = self.base_loss(images, labels)
        kd_loss = self.get_kd_loss(ref_data, feature_criterion=feature_criterion)

        return base_loss + ratio * kd_loss, {
            "base_loss": base_loss.item(),
            "kd_loss": kd_loss.item(),
        }

    def random_kd_loss(self, images, labels, batch_size, ratio, **_):
        random_noise = torch.rand(batch_size, *images.shape[1:]).to(images.device)
        return self.general_kd_loss(images, labels, random_noise, ratio)

    def lwf_loss(self, images, labels, ratio, **_):
        student_logits = self.model(images)
        base_loss = self.criterion(student_logits, labels)
        kd_loss = self.get_kd_loss(ref_data=images, student_logits=student_logits)

        return base_loss + ratio * kd_loss, {
            "base_loss": base_loss.item(),
            "kd_loss": kd_loss.item(),
        }

    def lwf_random_loss(self, images, labels, noise_ratio=0.5, ratio=0.2, **_):
        random_gaussian_noise = torch.randn(*images.shape, device=images.device)
        mix_images = (1 - noise_ratio) * images + noise_ratio * random_gaussian_noise
        return self.general_kd_loss(images, labels, mix_images, ratio)


class ZSCLTrainer(KDTrainer):
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

    def zscl_loss(self, images, labels, ratio, **_):
        ref_images, _ = self.get_ref_data(self.ref_loader)
        return self.general_kd_loss(
            images, labels, ref_images, ratio, feature_criterion=self.feature_criterion
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
        **_,
    ):
        zscl_loss, loss_dict = self.zscl_loss(
            images, labels, ratio_ref, feature_criterion=self.feature_criterion
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
    ):
        ref_images, _ = self.get_ref_data(self.ref_loader)
        base_loss = self.base_loss(images, labels)

        student_logits = self.model.get_features(ref_images)

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


def get_kd_trainer(model, dataloaders, config, teacher_models):
    if "ref_dataset" in config.method:
        train_transform, _ = load_transform()
        dataset_name, dataloader_config = (
            config.method.ref_dataset,
            config.method.ref_dataset_config,
        )

        dataloaders["ref"] = get_dataloader(
            dataset_name=dataset_name,
            root=config.data.root,
            mode=dataloader_config.split_name,
            transform=train_transform,
            seed=config.task.seed,
            **dataloader_config,
        )

    if config.method.name == "previous_aware_zscl":
        train_transform, _ = load_transform()

        previous_config = config.method.previous_config
        strategy = config.method.selected_strategy

        if strategy in DATASET_MAPPING:
            dataloader = get_dataloader(
                dataset_name=strategy,
                root=config.data.root,
                mode=previous_config.split_name,
                transform=train_transform,
                **previous_config,
            )
        else:
            if strategy == "learnable_input":
                previous_approximation = (
                    torch.load("learnable_input_with_gt_labels.pt").cpu().detach()
                )
            elif strategy == "random_noise":
                previous_approximation = torch.randn(
                    previous_config.sample_num, 3, 224, 224
                )
            elif strategy == "from_ref_dataset":
                dist, indices = inference_feature_distance(
                    teacher_models["pretrained"],
                    teacher_models["prev"],
                    dataloaders["ref"],
                )
                argmax_idx = dist.argsort(descending=True)[: previous_config.sample_num]

                previous_approximation = deepcopy(dataloaders["ref"].dataloader.dataset)
                previous_approximation._data_list = [
                    previous_approximation._data_list[indices[idx]]
                    for idx in argmax_idx
                ]
            elif strategy == "pgd_attack":
                previous_approximation = NoisyImageListDataset(
                    noise_path="outputs/adv_images/adv_images.pt",
                    image_list_path="largest_distance.txt",
                    transform=train_transform,
                )
                previous_approximation._data_list = [
                    previous_approximation._data_list[i] for i in range(32)
                ]

            dataloader = build_iter_dataloader(
                previous_approximation,
                **previous_config,
            )

        dataloaders["prev"] = dataloader

    arguments = [model, dataloaders, config, teacher_models]

    match config.method.name:
        case "zscl":
            return ZSCLTrainer(*arguments)
        case "previous_aware_zscl":
            return PreviousAwareZSCLTrainer(*arguments)
        case "mix_teacher":
            return MixTeacherKDTrainer(*arguments)
        case _:
            return KDTrainer(*arguments)
