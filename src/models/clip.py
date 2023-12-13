import open_clip
import torch
import torch.nn as nn
from tqdm import tqdm

from src.datasets.utils import load_class_name_list
from src.template import SIMPLE_TEMPLATE_LIST, ClassTemplate


class ClipBase(nn.Module):
    def __init__(self, model_name="ViT-B-16", pretrained="openai"):
        super().__init__()
        self.model = open_clip.create_model_from_pretrained(
            model_name,
            pretrained=pretrained,
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
    def __init__(
        self, clip_base, classification_head, freeze_classification_head=False
    ):
        super().__init__()
        self.clip_base = clip_base
        self.classification_head = classification_head
        self.freeze_classification_head = freeze_classification_head

        if freeze_classification_head:
            for p in self.classification_head.parameters():
                p.requires_grad = False

    @property
    def preprocess_config(self):
        return self.clip_base.preprocess_config

    def forward(self, images):
        return self.classification_head(self.clip_base(images))

    def get_params(self):
        clip_base_params = [
            p for p in self.clip_base.model.visual.parameters() if p.requires_grad
        ]

        if self.freeze_classification_head:
            return [{"params": clip_base_params}]

        classification_head_params = [
            p for p in self.classification_head.parameters() if p.requires_grad
        ]

        return [
            {"params": clip_base_params},
            {"params": classification_head_params},
        ]


# In PureClip model, the text-encoder is involved in the training progress.
class PureClip(nn.Module):
    def __init__(self, model_config, class_name_list):
        super().__init__()
        self.model = open_clip.create_model_from_pretrained(
            model_config.vit_base,
            pretrained=model_config.pretrained,
            return_transform=False,
        )

        self.tokenizer = open_clip.get_tokenizer(model_config.vit_base)
        self.template = SIMPLE_TEMPLATE_LIST[0]
        self.class_tokens = self.tokenizer(
            [self.template(t) for t in class_name_list]
        ).cuda()

    @property
    def preprocess_config(self):
        return self.model.visual.preprocess_cfg

    def forward(self, images, normalize=True):
        image_embeddings = self.model.encode_image(images, normalize=normalize)
        text_embeddings = self.model.encode_text(self.class_tokens, normalize=normalize)

        return self.model.logit_scale.exp() * image_embeddings @ text_embeddings.T

    def get_params(self):
        return [{"params": [p for p in self.model.parameters() if p.requires_grad]}]


def get_model(config, device="cuda", template_list=SIMPLE_TEMPLATE_LIST):
    class_name_list = load_class_name_list(config)

    model_config = config.model

    if model_config.get("use_pure_clip", False):
        return PureClip(model_config, class_name_list).to(device)

    model_name, pretrained = model_config.vit_base, model_config.pretrained

    if (model_name, pretrained) in open_clip.list_pretrained():
        clip_base = ClipBase(model_config.vit_base, model_config.pretrained).to(device)
    else:
        # TODO: check if freeze classification head
        clip_base = ClipBase(model_name)
        state_dict = torch.load(pretrained)["model"]
        clip_base.model.visual.load_state_dict(state_dict)
        clip_base.to(device)

    tokenizer = open_clip.get_tokenizer(model_name)

    class_template = ClassTemplate(clip_base.model, tokenizer, template_list, device)

    classification_head = ClassificationHead.initialize(class_name_list, class_template)

    return ClipClassifier(
        clip_base,
        classification_head,
        model_config.get("freeze_classification_head", False),
    ).to(device)


if __name__ == "__main__":
    from omegaconf import OmegaConf

    from src.datasets.core_dataset import Flowers102

    config = OmegaConf.load("config.yaml")
    dataset = Flowers102(root=config.data.root, mode="val")

    model = get_model(config, device="cuda")

    a = torch.rand(16, 3, 224, 224).cuda()
    print(model(a))
