# speaker-verification-test

Tested on Ubuntu 20.04, Python 3.10.6 with RTX A4500 GPU.

## Experiment reproducibility

The 'dataset_exp_1/' directory contains the dataset used in the first
experiments in the paper. 

The 'dataset_exp_2/' directory contains the
dataset used in the second experiments in the paper.

The 'dataset_exp_3/' directory contains the
dataset used in the third experiment, for the experiment it has (11labs, original(original folder contains copy of speakers audio from second experiment), rvc, tortoise, xtts) folders each one of them contains 20 speakers with 20 generated audio from the same text as original.

The 'dataset_exp_4/' directory contains the
dataset used in the fourth experiment, for the experiment it has (11labs, original(original folder contains copy of speakers audio from second experiment), rvc, tortoise, xtts) folders each one of them contains 20 speakers with 20 generated audio with but from a different text as original.

Files 'dataset_exp_1.pkl', 'dataset_exp_2.pkl', 'dataset_exp_3.pkl', 'dataset_exp_4.pkl'.  get dataset information respectively.
After running the experiments the
'scores_exp_1.csv' file will contain result of the first dataset experiment,
'scores_exp_2.csv' will contain result of the second dataset experiment.
'scores_exp_3.csv' will contain result of the third dataset experiment.
'scores_exp_4.csv' will contain result of the fourth dataset experiment.

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

Make sure to have a dataset in the `dataset_exp_1/`, `dataset_exp_2/`, `dataset_exp_3/`, `dataset_exp_4/` locations,
which contains subdirectories representing speakers, which
contain `.wav` or `.mp3` files.

The the experiments can be run using the following command:

```
python exp_1_1.py - to run first experiment for `dataset_exp_1/` files
python exp_1_2.py - to run second experiment for `dataset_exp_2/` files
python create_average_emb.py - to create average embedings for second experiment using `dataset_exp_1/` files (To run first experiment faster better to use before exp_1_1.py to cache all embedings from `dataset_exp_1/` dataset)
python exp_1_3&4.py - to run third and fourth experiment for `dataset_exp_3/` and `dataset_exp_4/` datasets.
```

The output of the program will be a CSV file `scores_exp_1.csv` for first experiment,
the `scores_exp_2.csv` for second experiment,
the `scores_exp_3.csv` for third experiment and
the `scores_exp_4.csv` for fourth experiment.
They containing all experiments scores and information.


