import open_clip
import torch
import torch.nn as nn
from tqdm import tqdm

from src.template import SIMPLE_TEMPLATE_LIST, ClassTemplate


class ClipBase(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.model = open_clip.create_model_from_pretrained(
            model_config.vit_base,
            pretrained=model_config.pretrained,
            return_transform=False,
        )

    @property
    def preprocess_config(self):
        return self.model.visual.preprocess_cfg

    def forward(self, images, normalize=True):
        return self.model.encode_image(images, normalize=normalize)


# take text embeddings as a linear layer
class ClassificationHead(nn.Linear):
    def __init__(self, weights):
        output_dim, input_dim = weights.shape
        super().__init__(input_dim, output_dim, bias=False)
        self.weight = torch.nn.Parameter(weights.clone())

    @classmethod
    def initialize(cls, class_name_list, class_template):
        classifier_weights = torch.stack(
            [
                class_template(class_name)
                for class_name in tqdm(class_name_list, desc="Build classifier weights")
            ],
            dim=0,
        )
        return cls(classifier_weights.detach().cpu())


class ClipClassifier(nn.Module):
    def __init__(self, clip_base, classification_head):
        super().__init__()
        self.clip_base = clip_base
        self.classification_head = classification_head

    @property
    def preprocess_config(self):
        return self.clip_base.preprocess_config

    def forward(self, images):
        return self.classification_head(self.clip_base(images))


def load_model(
    model_config, class_name_list, template_list=SIMPLE_TEMPLATE_LIST, device="cuda"
):
    clip_base = ClipBase(model_config).to(device)
    tokenizer = open_clip.get_tokenizer(model_config.vit_base)

    class_template = ClassTemplate(clip_base.model, tokenizer, template_list, device)

    classification_head = ClassificationHead.initialize(class_name_list, class_template)

    return ClipClassifier(clip_base, classification_head).to(device)


if __name__ == "__main__":
    from omegaconf import OmegaConf

    from src.datasets.core_dataset import Flowers102

    config = OmegaConf.load("config.yaml")
    dataset = Flowers102(root=config.data.root, mode="val")

    model = load_model(config.model, dataset.class_name_list, device="cuda")

    a = torch.rand(16, 3, 224, 224).cuda()
    print(model(a))
