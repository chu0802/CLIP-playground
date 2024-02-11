from abc import abstractmethod
from copy import deepcopy

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.datasets.utils import load_class_name_list
from src.template import SIMPLE_TEMPLATE_LIST, ClassTemplate


class ModelBase(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self):
        raise NotImplementedError

    @abstractmethod
    def get_params(self):
        raise NotImplementedError

    @abstractmethod
    def get_state_dict(self):
        raise NotImplementedError


class VisualClipBase(nn.Module):
    def __init__(self, visual_model):
        super().__init__()
        self.model = visual_model

    @property
    def preprocess_config(self):
        return self.model.preprocess_cfg

    def forward(self, images, normalize=True):
        features = self.model(images)
        return F.normalize(features, dim=-1) if normalize else features


class ClipBase(nn.Module):
    def __init__(self, model_name="ViT-B-16"):
        super().__init__()
        self.model = open_clip.create_model_from_pretrained(
            model_name,
            pretrained="openai",
            return_transform=False,
        )

    @property
    def preprocess_config(self):
        return self.model.visual.preprocess_cfg

    def forward(self, images, normalize=True):
        features = self.model.visual(images)
        return F.normalize(features, dim=-1) if normalize else features


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


class ClipClassifier(ModelBase):
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

    def get_prediction_from_features(self, feats):
        return self.classification_head(feats)

    def forward(self, images, get_features=False):
        features = self.clip_base(images)
        if get_features:
            return features
        return self.classification_head(features)

    def get_params(self):
        clip_base_params = [
            p for p in self.clip_base.model.parameters() if p.requires_grad
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

    def get_state_dict(self):
        if self.freeze_classification_head:
            return self.clip_base.model.state_dict()
        return self.state_dict()

    def load_state_dict(self, state_dict):
        if self.freeze_classification_head:
            self.clip_base.model.load_state_dict(state_dict)
        else:
            super().load_state_dict(state_dict)


# In PureClip model, the text-encoder is involved in the training progress.
class PureClip(ModelBase):
    def __init__(self, model_name, class_name_list, device="cuda"):
        super().__init__()
        self.model = open_clip.create_model_from_pretrained(
            model_name,
            pretrained="openai",
            return_transform=False,
        )

        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.template = SIMPLE_TEMPLATE_LIST[0]
        self.device = device
        self.class_tokens = self.tokenize(class_name_list)

    @property
    def preprocess_config(self):
        return self.model.visual.preprocess_cfg

    def tokenize(self, texts, device="cuda"):
        return self.tokenizer([self.template(t) for t in texts]).to(device)

    def encode(self, images=None, text=None, normalize=True):
        if images is None:
            return self.model.encode_text(text, normalize=normalize)
        if text is None:
            return self.model.encode_image(images, normalize=normalize)

    # to fit the format of clip-classifier, we send a list of data to pure-clip if text is neeeded.
    def forward(self, images, text=None, normalize=True, get_features=False):
        if text is None:
            text = self.class_tokens

        image_embeddings = self.encode(images=images, normalize=normalize)
        text_embeddings = self.encode(text=text, normalize=normalize)

        if get_features:
            return image_embeddings, text_embeddings, self.model.logit_scale.exp()

        return self.model.logit_scale.exp() * image_embeddings @ text_embeddings.t()

    def get_params(self):
        exclude_param = "logit_scale"
        return [
            {
                "params": [
                    p
                    for k, p in self.model.named_parameters()
                    if p.requires_grad and exclude_param not in k
                ]
            }
        ]

    def get_state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)


def build_classification_head(model_config, class_name_list, template_list):
    # We must send clip base model to cuda in this step since we are inferencing the outputs of the class names.
    # However, the built classification head in this step haven't been sent to cuda yet.
    clip_base = ClipBase(model_config.vit_base).to("cuda")
    tokenizer = open_clip.get_tokenizer(model_config.vit_base)
    class_template = ClassTemplate(clip_base.model, tokenizer, template_list, "cuda")
    classification_head = ClassificationHead.initialize(class_name_list, class_template)

    del clip_base

    return classification_head


def get_model(
    config,
    pretrained=False,
    freeze=False,
    template_list=SIMPLE_TEMPLATE_LIST,
    device="cuda",
):
    class_name_list = load_class_name_list(config)

    model_config = config.model

    # first initialize a clip model pre-trained by openai
    if model_config.get("use_pure_clip", False):
        model = PureClip(model_config.vit_base, class_name_list, device=device)
    else:
        classification_head = build_classification_head(
            model_config, class_name_list, template_list
        )
        model = ClipClassifier(
            VisualClipBase(ClipBase(model_config.vit_base).model.visual),
            classification_head,
            model_config.get("freeze_classification_head", False),
        )

    # then load from a checkpoint if not pre-trained
    if model_config.pretrained != "openai" and not pretrained:
        model.load_state_dict(torch.load(model_config.pretrained)["model"])

    model = model.to(device)

    if freeze:
        for _, v in model.named_parameters():
            v.requires_grad = False
        model.eval()

    if config.task.get("distributed", False) and not freeze:
        model = nn.parallel.DistributedDataParallel(model)

    return model


if __name__ == "__main__":
    from omegaconf import OmegaConf

    from src.datasets.core_dataset import Flowers102

    config = OmegaConf.load("configs/split_teacher_config.yaml")
    dataset = Flowers102(root=config.data.root, mode="val")

    model = get_model(config, pretrained=True, freeze=True, device="cuda")

    a = torch.rand(16, 3, 224, 224).cuda()
    print(model(a))
