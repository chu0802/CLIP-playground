import datetime
import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

import wandb
from src.trainer.optimizer import get_optimizer
from src.trainer.scheduler import CosineLRScheduler
from src.utils import AccuracyMeter, dump_config


class Trainer:
    def __init__(self, model, dataloaders, config):
        self.model = model
        self.dataloaders = dataloaders
        self.config = config
        self.criterion = nn.CrossEntropyLoss()

        if self.training_mode:
            self.optimizer = get_optimizer(self.model, self.config.task)
            self.lr_scheduler = CosineLRScheduler(
                self.optimizer, self.config.task, self.num_total_train_steps
            )

        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        self.output_dir = (
            Path(self.config.task.output_dir)
            / self.config.model.vit_base
            / self.config.data.name
            / timestamp
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.local_log = defaultdict(dict)

        dump_config(self.config, self.output_dir / "config.json")

    def save(self, epoch):
        # TODO: check if freeze classification head or not
        visual_state_dict = self.model.clip_base.model.visual.state_dict()

        save_obj = {"model": visual_state_dict}

        save_path = self.output_dir / f"checkpoint_{epoch}.pth"

        print(f"Saving checkpoint at epoch {epoch} to {save_path}.")
        torch.save(save_obj, save_path)

    @property
    def method_config(self):
        return self.config.model.method

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

    def logging(self, local_desc=None, **message_dict):
        wandb.log(message_dict)
        if local_desc is not None:
            self.local_log[local_desc].update(message_dict)

    def dump_results(self, filename="results.json"):
        with open(self.output_dir / filename, "w") as f:
            json.dump(self.local_log, f, indent=4)

    def base_loss(self, images, labels, **_):
        outputs = self.model(images)
        return self.criterion(outputs, labels)

    def train_step(self, images, labels):
        # need to step lr_scheduler first since in this repo I didn't explictly set a learning rate in the optimizer.
        self.lr_scheduler.step()

        loss_fn = getattr(self, f"{self.method_config.name}_loss")
        loss = loss_fn(images, labels, **self.method_config.params)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item()

    def evaluate(self, dataloader=None):
        if dataloader is None:
            dataloader = self.test_loader

        self.model.eval()

        scores = AccuracyMeter()

        dataloader.init()
        with torch.no_grad(), tqdm(total=len(dataloader)) as pbar:
            for images, labels in dataloader:
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

                for i, (images, labels) in enumerate(self.train_loader):
                    loss = self.train_step(images, labels)

                    pbar.set_postfix_str(f"lr: {self.lr:.2e}, loss: {loss:.2e}")
                    pbar.update(1)

                    if i % self.log_interval == 0:
                        self.logging(lr=self.lr, loss=loss)

                if self.val_loader and set_validation:
                    self.logging(val_acc=self.evaluate(self.val_loader))

                self.save(epoch)


class KDTrainer(Trainer):
    def __init__(self, model, dataloaders, config, teacher_model):
        super().__init__(model, dataloaders, config)
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        self.kl_criterion = nn.KLDivLoss()

    def random_kd_loss(self, images, labels, batch_size, ratio, **_):
        base_loss = self.base_loss(images, labels)

        random_noise = torch.rand(batch_size, *images.shape[1:]).to(images.device)

        with torch.no_grad():
            teacher_logits = self.teacher_model(random_noise)

        student_logits = self.model(random_noise)

        soft_labels = nn.functional.softmax(teacher_logits, dim=-1)
        soft_preds = nn.functional.log_softmax(student_logits, dim=-1)

        kd_loss = -(soft_labels * soft_preds).sum() / soft_preds.shape[0]

        return base_loss + ratio * kd_loss

    def lwf_loss(self, images, labels, ratio, **_):
        student_logits = self.model(images)
        base_loss = self.criterion(student_logits, labels)

        with torch.no_grad():
            teacher_logits = self.teacher_model(images)

        soft_labels = nn.functional.softmax(teacher_logits, dim=-1)
        soft_preds = nn.functional.log_softmax(student_logits, dim=-1)

        kd_loss = -(soft_labels * soft_preds).sum() / soft_preds.shape[0]

        return base_loss + ratio * kd_loss

    def lwf_random_loss(self, images, labels, noise_ratio=0.5, ratio=0.2, **_):
        base_loss = self.base_loss(images, labels)

        random_noise = torch.rand(*images.shape).to(images.device)

        mix_images = (1 - noise_ratio) * images + noise_ratio * random_noise

        with torch.no_grad():
            teacher_logits = self.teacher_model(mix_images)

        student_logits = self.model(mix_images)

        soft_labels = nn.functional.softmax(teacher_logits, dim=-1)
        soft_preds = nn.functional.log_softmax(student_logits, dim=-1)

        kd_loss = -(soft_labels * soft_preds).sum() / soft_preds.shape[0]

        return base_loss + ratio * kd_loss
