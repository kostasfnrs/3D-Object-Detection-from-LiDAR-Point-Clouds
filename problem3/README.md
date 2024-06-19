# Instructions for Running Problem 3 Experiments

In Problem 3, we have kept the reproduction process and the code straightforward. We haven't altered the main training files; the changes are confined to `model.py` and `config.yaml`.

## How to Run the Experiments

We conducted four different experiments, as detailed in our report. These include:
- Baseline
- Large Baseline
- Large Baseline + Attention
- Baseline + Attention

To replicate these experiments, you need to rename `model.py` and `config.yaml` appropriately. In our submission, we have included two versions for each. For the model, there are: `model_original.py` and `model_transformer.py`. For the config, we have: `config_original.yaml` and `config_large.yaml`. 

To run an experiment, rename the files as outlined below. For example, change `model_original.py` to `model.py` and `config_large.yaml` to `config.yaml`, while keeping the other files unchanged.

Follow these steps to recreate the experiments:
- Baseline -> Rename: `config_original.yaml` to `config.yaml` and `model_original.py` to `model.py`.
- Large Baseline -> Rename: `config_large.yaml` to `config.yaml` and `model_original.py` to `model.py`.
- Large Baseline + Attention -> Rename: `config_large.yaml` to `config.yaml` and `model_transformer.py` to `model.py`.
- Baseline + Attention -> Rename: `config_original.yaml` to `config.yaml` and `model_transformer.py` to `model.py`.

## WandB Results
You can find the Weights & Biases results in the file named `wandb_export_team07.csv`.
