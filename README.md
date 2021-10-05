![Tests](https://github.com/bayer-science-for-a-better-life/molnet-geometric-lightning/actions/workflows/python-package-conda.yml/badge.svg) 

# molnet-geometric-lightning

This is a package for benchmarking the [MoleculeNet datasets](https://pubs.rsc.org/en/content/articlelanding/2018/sc/c7sc02664a) present in the [Open Graph Benchmark](https://ogb.stanford.edu/) on different [graph convolutional neural network](https://distill.pub/2021/gnn-intro/) architectures.
The neural networks are implemented using [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) and [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning).
Large parts of this code are borrowed from PyTorch Geometric and OGB examples, therefore this package is available under the same license (MIT).

## Why?

The OGB library offers premade data objects compatible with PyTorch Geometric.
While convenient, this makes it difficult to implement different featurizations.
Furthermore, the PyTorch Lightning framework makes for easier-to-maintain code, with a nice command line interface and Tensorboard logging built-in.

## Installation

After cloning this repo, you should be able to install with:

```conda env create```
```conda activate mgl```

Note: depending on your hardware, you may need to install the CUDA toolkit as well.
For instance, you might have to add a line `- cudatoolkit=10.2` to `environment.yml`.

## Example Usage

The following will train 5 models on the `bbbp` dataset with the default parameters.
The models will be stored in `example_models/`, and the data will be downloaded to `datasets/`.

```shell script
 python molnet_geometric_lightning/train.py --default_root_dir=example_model/ --dataset_name=bbbp --dataset_root=datasets/ --gpus=1 --max_epochs=100 --n_runs=5 
```

Replace the directories to your liking, and `bbbp` with any name from MoleculeNet, for example `tox21`, `muv`, `hiv`, `pcba`, `bace`, `esol`.

## Model evaluation

Validation curves and test set performance are logged to `default_root_dir`.
Start a Tensorboard server with `default_root_dir` as the log directory.
From the above example, something like:

```shell script
tensorboard --logdir=/full/path/to/example_model/
```