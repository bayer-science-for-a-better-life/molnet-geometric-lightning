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

```
conda env create
conda activate mgl
```

Then, install this package with:

```pip install -e .```

Then, install this package with:

```pip install -e .```

Note: depending on your hardware, you may need to install the CUDA toolkit as well.
For instance, you might have to add a line `- cudatoolkit=10.2` to `environment.yml`.

## Benchmarks

For instructions on recreating the paper benchmarks, see the `notebooks` directory.
You can install jupyter with `pip install jupyter`.

## Example command-line usage

The following will train 5 models on the `bbbp` dataset with the default parameters.
The models will be stored in `example_models/`, and the data will be downloaded to `datasets/`.

```shell script
 python molnet_geometric_lightning/train.py --default_root_dir=example_model/ --dataset_name=bbbp --dataset_root=datasets/ --gpus=1 --max_epochs=100 --n_runs=5 
```

Replace the directories to your liking, and `bbbp` with any name from MoleculeNet, for example `tox21`, `muv`, `hiv`, `pcba`, `bace`, `esol`.

## Using this repository as a template

This repository is intended to be used as a template for other experiments.
Don't fork it!
Instead use the "Use this template" button at the top.
This "forks" the code without the full commit history.

In addition to changing the dataset and model code, there are some other things you should update to get the most out of this template:

- [ ] Update the package name and imports: this means replacing `molnet_geometric_lightning` in filenames **and files** with the name of your package.
- [ ] Update `setup.py`: this means changing `molnet-geometric-lightning` in `setup.py` to your package name.
- [ ] Update `train.py`: you might add new parameters that need to be reflected here.
- [ ] Update `test_integration`: you should modify the arguments here to make sure your modified code is tested. You get built-in Github CI for free!
- [ ] Update `environment.yml`: in addition to any extra packages you need, don't forget to change the environment name.
- [ ] Update `README.md`: should be a no-brainer. In particular, don't forget to change the badge at the top of the `README.md` file!

By default, the CI checks code formatting.
This can be annoying if you don't want to spend time making your code Flake8 compliant.
To stop this, you can delete the `Flake8` section in `.github/workflows/python-package-conda.yml`

## Model evaluation

Validation curves and test set performance are logged to `default_root_dir`.
Start a Tensorboard server with `default_root_dir` as the log directory.
From the above example, something like:

```shell script
tensorboard --logdir=/full/path/to/example_model/
```
