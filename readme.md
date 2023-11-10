
# Background
This repository is the code base for a project completed for the course ELEC5305 "Acoustics, Speech and Signal Processing" at the University of Sydney.


# Setup

## Python
After setting up your python environment, install the following packages:

´´´
pip3 install torch torchvision torchaudio pandas numpy matplotlib ipykernel pillow torchtoolbox scikit-learn tqdm timm seaborn
´´´

## Datasets

Two datasets are required to run this code. The are available at https://www.kaggle.com/datasets/colinmerk/swissbirds/data and https://www.kaggle.com/datasets/christofhenkel/birdclef2021-background-noise.


## Configure Code

The notebook in train_notebook_grid_training.ipynb contains a flag whether the code is run on Kaggle or locally. This flag toggles between the paths and devices. Depending on where you run it, you need to adapt the paths for the datasets, as they are global paths.
