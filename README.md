# speaker-verification-test

Tested on Ubuntu 20.04, Python 3.10.6 with RTX A4500 GPU.

## Experiment reproducibility

The `dataset.pkl` file contains prepared dataset, which was used to
get the scores for experiment results in the paper. In current version,
this file should be in the root folder of the project, so it is read 
by the `main.py` script instead of generating new voice pairs randomly.

The `scores.csv` file contains experiment results when running the experiment
with the provided `dataset.pkl` file.

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
