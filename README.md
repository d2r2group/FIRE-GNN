<div align="center">
  <img alt="FIRE-GNN Logo" src=logo.svg width="200"><br>
</div>

# `FIRE-GNN` – Force-Informed, Relaxed Equivariance Graph Neural Network

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16920229.svg)](https://doi.org/10.5281/zenodo.16920229)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

FIRE-GNN is a force-informed, relaxed equivariant graph neural network for predicting surface work functions and cleavage energies 
from slab structures. By incorporating surface-normal symmetry breaking and machine learning interatomic potential–derived force information, 
the approach achieves state-of-the-art accuracy and enables rapid, generalizable screening critical for the discovery of materials with tuned 
surface properties across the periodic table.

Please, cite the following [paper](https://doi.org/10.1002/aidi.202500162) if you use the architecture/model in your research:
```
@article{Hsu2025Dec,
	author = {Hsu, Circe and Schlesinger, Claire and Mudaliar, Karan and Leung, Jordan and Walters, Robin and Schindler, Peter},
	title = {{FIRE-GNN: Force-informed, Relaxed Equivariance Graph Neural Network for Rapid and Accurate Prediction of Surface Properties}},
	journal = {Advanced Intelligent Discovery},
	volume = {0},
	number = {0},
	pages = {0},
	year = {2025},
	month = dec,
	publisher = {Wiley},
	doi = {10.1002/aidi.202500162}
}
```

## Installation

To install via pip, you can create your own pip environment or use your global environment and install the requirements.txt file. 
```
cd FIRE
python -m venv firegnn
source firegnn/bin/activate
pip install -r requirements.txt
```

To install via micromamba (or any conda)
```
cd FIRE
micromamba env create -f environment.yml
micromamba activate firegnn
```

## Usage

In order to run FIRE-GNN, set up your slabs as a cif file, folder of cif files, or json list of cif strings. Run 
```
python run_custom.py FIRE-GNN-Model_structureid_split configs/struct_test_forces.json slabs.json
```