def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def _merge(theta0, theta1, alpha=0.5):
    return {
        key: alpha * theta0[key] + (1 - alpha) * theta1[key] for key in theta0.keys()
    }


def wise_ft(pretrained_model, finetuned_model, alpha=0.5):
    pretrained_weights = {k: v.clone() for k, v in pretrained_model.named_parameters()}
    finetuned_weights = {k: v.clone() for k, v in finetuned_model.named_parameters()}

    finetuned_model.load_state_dict(
        _merge(pretrained_weights, finetuned_weights, alpha=alpha)
    )

    return finetuned_model
