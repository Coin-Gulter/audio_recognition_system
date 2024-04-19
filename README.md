# speaker-verification-test

Tested on Ubuntu 20.04, Python 3.10.6 with RTX A4500 GPU.

## Experiment reproducibility

The 'dataset_exp_1/' directory contains the dataset used in the first
experiments in the paper. The 'dataset_exp_2/' directory contains the
dataset used in the second experiments in the paper.
Files 'dataset_exp_1.pkl' and 'dataset_exp_2.pkl' get dataset information respectively.
After running the experiments, the 'scores_exp_1.csv' file will contain experiment
results of the first dataset experiment and 'scores_exp_2.csv' will contain experiment
results of the second dataset experiment.
## Get dataset

The recommended way to download the dataset using git.
**Important!** Make sure that `git-lfs` installed!

On Ubuntu:

```
sudo apt install git-lfs
```

Then run:

```
git lfs install
git clone https://huggingface.co/datasets/vbrydik/ua-polit-tiny ./dataset
```

Alternatively, this dataset can be downloaded from the following sources: 
    - [Huggingface](https://huggingface.co/datasets/vbrydik/ua-polit-tiny).
    - [Google Drive](...)
    
*(TODO: Add dataset DOI.)*

## Environment setup

Set up virtual env:

```
python -m venv venv
source venv/bin/activate
```

Install modules

```
pip install -r requirements.txt
```

## Run experiments

Make sure to have a dataset in the 'dataset_exp_1/' and 'dataset_exp_2/' locations,
which contains subdirectories representing speakers, which
contain `.wav` files.

The the experiments can be run using the following command:

```
python exp_1_1.py - to run first experiment for 'dataset_exp_1/' files
python exp_1_2.py - to run second experiment for 'dataset_exp_2/' files
python create_average_emb.py - to create average embedings for second experiment using 'dataset_exp_1/' files (To run first experiment faster better to use before exp_1_1.py to cache all embedings from 'dataset_exp_1/' dataset)
```

The output of the program will be a CSV file `scores_exp_1.csv` for first experiment,
snd int the `scores_exp_2.csv` for second experiment.
They containing all experiments scores and information.
