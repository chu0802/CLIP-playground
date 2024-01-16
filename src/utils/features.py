import torch
from tqdm import tqdm


def inference_feature_distance(pretrained_model, finetuned_model, dataloader):
    dataloader.init()

    distance = []
    indices = []

    pretrained_model.eval()
    finetuned_model.eval()

    with torch.no_grad():
        for data in tqdm(
            dataloader, desc="Inference feature distance", total=len(dataloader)
        ):
            if isinstance(data, (list, tuple)):
                index = data[-1]
                # dataset with noise
                if len(data) == 4:
                    data = data[0] + data[1]
                else:
                    data = data[0]

            finetuned_features = finetuned_model.get_features(data)
            pretrained_features = pretrained_model.get_features(data)

            distance.append(torch.norm(finetuned_features - pretrained_features, dim=1))
            indices.append(index)

    return torch.cat(distance).detach().cpu(), torch.cat(indices).detach().cpu()
