# speaker-verification-test

Tested on Ubuntu 20.04, Python 3.10.6 with RTX A4500 GPU.

## Experiment reproducibility

The 'dataset_experiment_1/' directory contains the dataset used in the first
experiments in the paper. The 'dataset_experiment_2/' directory contains the
dataset used in the second experiments in the paper.
Files 'dataset_experiment_1.pkl' and 'dataset_experiment_2.json' get dataset information respectively
After running the experiments, the 'scores_experiment_1.csv' file will contain experiment
results of the first dataset experiment and 'scores_experiment_2.csv' will contain experiment
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

Make sure to have a dataset in the `./dataset` location,
which contains subdirectories representing speakers, which
contain `.wav` files.

The the experiments can be run using the following command:

```
python main.py
```

The output of the program will be a CSV file `scores.csv`,
containing all experiments scores and information.
