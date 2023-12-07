import torch

from .pre_define_template import OPENAI_IMAGENET_TEMPLATE_LIST, SIMPLE_TEMPLATE_LIST


class ClassTemplate:
    def __init__(self, model, tokenizer, template_list, device):
        self.template_list = template_list
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def __call__(self, class_name):
        self.model.eval()

        tokens = self.tokenizer(
            [template(class_name) for template in self.template_list]
        ).to(self.device)
        text_features = self.model.encode_text(tokens, normalize=True)
        text_features = text_features.mean(dim=0)
        text_features /= text_features.norm()

        return text_features * self.model.logit_scale.exp()
