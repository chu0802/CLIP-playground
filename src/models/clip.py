import open_clip
import torch


class ClipModel:
    def __init__(self, config):
        self.clip_base = open_clip.create_model_from_pretrained(
            config.vit_base, pretrained=config.pretrained, return_transform=False
        )
        self.tokenizer = open_clip.get_tokenizer(config.vit_base)

    @torch.no_grad()
    @torch.cuda.amp.autocast()
    def inference(self, image, label_space):
        label_space = self.tokenizer(label_space).cuda()

        image_features = self.clip_base.encode_image(image)
        text_features = self.clip_base.encode_text(label_space)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        return text_probs
