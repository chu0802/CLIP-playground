import datetime
import json
from collections import defaultdict
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

    def train_step(self, images, labels):
        # need to step lr_scheduler first since in this repo I didn't explictly set a learning rate in the optimizer.
        self.lr_scheduler.step()

        outputs = self.model(images)
        loss = self.criterion(outputs, labels)

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
