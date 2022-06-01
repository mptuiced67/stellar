# Annotation of Spatially Resolved Single-cell Data with STELLAR

PyTorch implementation of STELLAR, a geometric deep learning tool for cell-type discovery and identification in spatially resolved single-cell datasets. STELLAR takes as input annotated reference spatial single-cell dataset in which cells are assigned to their cell types, and unannotated spatial dataset in which cell types are unknown. STELLAR then generates annotations for the unannotated dataset. For a detailed description of the algorithm, please see our manuscript [Annotation of Spatially Resolved Single-cell Data with STELLAR](https://www.biorxiv.org/content/10.1101/2021.11.24.469947v1.full.pdf).


<p align="center">
<img src="https://github.com/snap-stanford/stellar/blob/main/images/stellar_overview.png" width="1100" align="center">
</p>


### Installation


**1. Python environment (Optional):**
We recommend using Conda package manager

```bash
conda create -n stellar python=3.8
source activate stellar
```

**2. Pytorch:**
Install [PyTorch](https://pytorch.org/). 
We have verified under PyTorch 1.9.1. For example:
```bash
conda install pytorch cudatoolkit=11.3 -c pytorch
```

**3. Pytorch Geometric:**
Install [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html), 
follow their instructions. We have verified under Pyg 2.0. For example:
```bash
conda install pyg -c pyg
```

**4. Other dependencies:**

Please run the following command to install additional packages that are provided in [requirements.txt](https://github.com/snap-stanford/stellar/blob/main/requirements.txt).
```bash
pip install -r requirements.txt
```

**Note:** We tested STELLAR with NVIDIA GPU, Linux, Python3. In particular, on Ubuntu 16.04 with NVIDIA Geforce 2080 Ti GPU and 1T CPU memory. We additionally tested the code on macOS (Intel chip).

### Getting started

We implemented STELLAR model in a self-contained class. To make an instance and train STELLAR:

```
stellar = STELLAR(args, dataset)
stellar.train()
_, results = stellar.pred()
```
### Datasets

CODEX multiplexed imaging datasets from healthy human tonsil and Barrett’s esophagus data are made available at [dryad](https://datadryad.org/stash/share/1OQtxew0Unh3iAdP-ELew-ctwuPTBz6Oy8uuyxqliZk). Our demo code assumes the data to be put under the folder `./data/ `.

### Demo

We provide several training examples with this repo:

- To run STELLAR on the CODEX healthy intestine data

```bash
python STELLAR_run.py --dataset Hubmap --input-dim 48 --num-heads 22
```

- To run STELLAR on the CODEX tonsil and BE data:

```bash
python STELLAR_run.py --dataset TonsilBE --input-dim 44 --num-heads 13 --num-seed-class 3
```
Memory usage and time:
-  Graph construction takes 32G physical memory for the HuBMAP dataset and 256G for Tonsil/BE. The longest construction takes around 10 minutes.
-  Running the algorithm on GPU takes less then 5G memory for both datasets and can finish within a few minutes.

We also provided a jupyter notebook [demo.ipynb](https://github.com/snap-stanford/stellar/blob/a556b5ef4fe43c512ccf092c1d06d73034dc8d4d/demo.ipynb) that shows example of running STELLAR on a downsampled dataset. Please consider to downsample more if there is a memory issue, but note that the performance of the model would degrade as the training data gets less. For users with limited memory and potentially limited access to GPU, please set the ``use-processed-graph`` to True to load pre-processsed data and can finish with CPU in about 30 mins.

### Use your own dataset

To use your own dataset, you just need to initialize [GraphDataset](https://github.com/snap-stanford/stellar/blob/a556b5ef4fe43c512ccf092c1d06d73034dc8d4d/datasets.py#L77) and give it to the input to our [stellar function](https://github.com/snap-stanford/stellar/blob/main/STELLAR.py).

```
dataset = GraphDataset(labeled_X, labeled_y, unlabeled_X, labeled_edges, unlabeled_edges)
stellar = STELLAR(args, dataset)
```

- labeled_X and unlabeled_X are node features matrices for the annotated reference dataset and target unannotated dataset, respectively. They should  have a shape [num_nodes, num_node_features] 
- labeled_y defines annotations for the annotated reference dataset with shape [num_nodes,] 
- labeled_edges and unlabeled_edges define the input graphs for the annotated reference dataset and target unannotated dataset, respectively. They should have a shape [2, num_edges] and they define edges of the graph

Example for HuBMAP dataset is shown in [load_hubmap_data](https://github.com/snap-stanford/stellar/blob/a556b5ef4fe43c512ccf092c1d06d73034dc8d4d/datasets.py#L30) function, and for Tonsil/BE dataset in [load_tonsilbe_data](https://github.com/snap-stanford/stellar/blob/a556b5ef4fe43c512ccf092c1d06d73034dc8d4d/datasets.py#L53).


### Citing

If you find our code and research useful, please consider citing:

```
@article{stellar2021,
  title={Annotation of Spatially Resolved Single-cell Data with STELLAR},
  author={Brbic, Maria and Cao, Kaidi and Hickey, John W and Tan, Yuqi and Snyder, Michael and Nolan, Garry P and Leskovec, Jure},
  journal={bioRxiv},
  year={2021},
}
```
