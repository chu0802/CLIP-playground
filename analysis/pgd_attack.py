from pathlib import Path

import torch
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

import wandb
from analysis.utils import TEST_CONFIG, get_model
from src.datasets.base import ImageListDataset
from src.datasets.transform import load_transform
from src.datasets.utils import build_iter_dataloader


class PGDAttacker:
    def __init__(
        self, fine_model, pre_model, dataloader, eps=16 / 255, alpha=2 / 255, steps=32
    ):
        self.fine_model = fine_model
        self.pre_model = pre_model
        self.dataloader = dataloader
        self.eps = eps
        self.alpha = alpha
        self.steps = steps

    def loss(self, images, adv_images):

        # step 1: min( d ( student(A) , teacher(B) ) )
        # student_org_image_features = self.fine_model.get_features(images)
        # student_adv_image_features = self.fine_model.get_features(adv_images)

        # anchor_loss = torch.norm(student_org_image_features - student_adv_image_features, dim=1).mean()

        # step 2: max( d ( student(B), teacher(B) ) )

        teacher_adv_images_features = self.pre_model.get_features(adv_images)
        student_adv_images_features = self.fine_model.get_features(adv_images)

        diverse_loss = -torch.norm(
            student_adv_images_features - teacher_adv_images_features, dim=1
        ).mean()

        return diverse_loss, {"diverse_loss": diverse_loss.item()}
        # return anchor_loss + diverse_loss, {"anchor_loss": anchor_loss.item(), "diverse_loss": diverse_loss.item()}

    def evaluation(self, noise_images):
        dist = []
        for adv_image in noise_images:
            student_output = self.fine_model.get_features(adv_image)
            teacher_output = self.pre_model.get_features(adv_image)

            dist.append(torch.norm(student_output - teacher_output, dim=1))
        dist = torch.cat(dist, dim=0).detach().cpu()
        print(dist)
        print(dist.mean().item())

    def train_step(self, images, adv_images):
        # return images
        for _ in tqdm(range(self.steps)):
            adv_images.requires_grad = True
            loss, loss_dict = self.loss(images, adv_images)
            self.logging(**loss_dict)
            # Update adversarial images
            grad = torch.autograd.grad(
                loss, adv_images, retain_graph=False, create_graph=False
            )[0]

            # change the sign here to minus, by default we'd like to minimize the derived loss above.
            adv_images = adv_images.detach() - self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images

    def train(self, save_dir=None):
        self.dataloader.init()

        noise, noise_images = [], []
        for images, _, _ in self.dataloader:
            images = images.clone().detach()
            adv_images = images.clone().detach()
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()
            # ret_images =  adv_images.clone().detach()
            ret_images = self.train_step(images, adv_images)

            noise.append(ret_images - images)
            noise_images.append(ret_images)

        self.evaluation(noise_images)

        noise = torch.cat(noise, dim=0)
        noise_images = torch.cat(noise_images, dim=0)

        if save_dir:
            save_dir.mkdir(exist_ok=True, parents=True)
            # torch.save(torch.zeros_like(noise).detach().cpu(), save_dir / "adv_images.pt")
            torch.save(noise.detach().cpu(), save_dir / "adv_images.pt")
            self.save(noise_images, save_dir=save_dir)

    def save(self, images, save_dir):
        (save_dir / "1").mkdir(exist_ok=True, parents=True)
        for i, image in enumerate(images):
            save_image(image, save_dir / "1" / f"{i:02d}.jpg")

    def logging(self, use_wandb=True, **message_dict):
        if use_wandb:
            wandb.log(message_dict)


def main(config):
    fine_model = get_model(config, device="cuda", freeze=True)
    pretrained_model = get_model(config, device="cuda", freeze=True, pretrained=True)

    train_trans, eval_trans = load_transform()

    custom_trans = transforms.Compose(
        [
            transforms.CenterCrop(224),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
        ]
    )

    dataset = ImageListDataset("largest_distance.txt", transform=train_trans)

    dataloader = build_iter_dataloader(
        dataset=dataset,
        batch_size=32,
        shuffle=False,
        drop_last=False,
        device="cuda",
    )

    attacker = PGDAttacker(fine_model, pretrained_model, dataloader=dataloader)

    attacker.train(save_dir=Path("outputs/adv_images"))


if __name__ == "__main__":
    wandb.init(
        project="pgd_attack",
        name="pgd_attack",
    )
    main(TEST_CONFIG)
    wandb.finish()
