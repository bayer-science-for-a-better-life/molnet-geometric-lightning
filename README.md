# Solubility and H-bonding

Uses rdkit to add hydrogen bonds as edges for graph convolutional models.

## Installation

Tested on `dgx-02-**`. After pulling the repo:

```conda env create```

## Example Usage

Works for all datasets in MoleculeNet.
Be warned: generating all hbonds for a large dataset can take days.

With h-bonds:

```shell script
 python solubility/train.py --default_root_dir=/home/gltyk/hbond_runs/bbbp/hbond --dataset_name=bbbp --dataset_root=/home/gltyk/bbbp_hbonds/ --hbonds --gpus=1 --max_epochs=100 --num_sanity_val_steps=0 
```

without h-bonds:

```shell script
 python solubility/train.py --default_root_dir=/home/gltyk/hbond_runs/bbbp/baseline --dataset_name=bbbp --dataset_root=/home/gltyk/bbbp/ --gpus=1 --max_epochs=100 --num_sanity_val_steps=0
```

Replace the directories to your liking, and `bbbp` with any name from MoleculeNet, for example `tox21`, `muv`, `hiv`, `pcba`, `bace`, `esol`.

## Model evaluation

Validation curves and test set performance are logged to `default_root_dir`.
Start a Tensorboard server with `default_root_dir` as the log directory.