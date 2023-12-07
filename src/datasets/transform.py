from open_clip.transform import PreprocessCfg, image_transform_v2


def load_model_transform(model_preprocess_config):
    pp_cfg = PreprocessCfg(**model_preprocess_config)

    train_transform = image_transform_v2(
        pp_cfg,
        is_train=True,
    )
    eval_transform = image_transform_v2(
        pp_cfg,
        is_train=False,
    )

    return train_transform, eval_transform
