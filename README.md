# Mix-Teacher Memory Retrieval for Continual Learning

The official Pytorch implementation of "Mix-Teacher Memory Retrieval for Continual Learning".

## Usage

This repo supports 2 scripts for easy access, every script is modulized:

### Training and evaluate on all datasets
```bash
python -m scripts.train_and_eval --config_path <CONFIG_PATH> --pretrained_dataset <PRETRAINED_DATASET> --dataset <DATASET>
```
### Continually training

```bash
python -m scripts.continually_train --config_path <CONFIG_PATH> --dataset_seq <DATASET_SEQ>
```

*Note: `<DATASET_SEQ>` shouuld be a string with several dataset splitted by comma.*
