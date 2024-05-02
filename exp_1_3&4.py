import os
import tqdm
import timeit
import numpy as np
import pandas as pd
import pickle as pkl
import json
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

from core import (
    Pipeline,
    Pyannote,
    WavLM,
    TitaNet,
    Ecapa
)
from core.metrics import (
    compute_eer, 
    compute_min_dcf,
    compute_far_frr,
)
from core.audio import Audio
from core.normalize import normalize
from utils import make_dataset, make_dataset_3_4



def get_pipeline(name: str):
    if name == "ecapa":
        return Ecapa()
    elif name == "titanet":
        return TitaNet()



def get_label(emb_name: str, file: str) -> int:
    """
    Return 0 if different speakers, 1 if same speakers.
    """
    return int(emb_name.split("/")[-1] == file.split("/")[-2])


def get_value_from_csv(filename, key, value):
    """
    This function opens a CSV file, finds the value from the specified key row and value column,
    and returns it.
    """

    df = pd.read_csv(filename)

    return df[df['pipeline'] == key][value]


def evaluate_pipeline(
    pipeline, 
    data, 
) -> pd.DataFrame:

    scores = []
    labels = []
    embedings = []
    elapsed_time = []

    with open(f'./avg_embedings/{pipeline.name}.json', '+r') as f:
        embedings = json.load(f)

    df = pd.read_csv("scores_exp_1.csv")
    thresh = float(df[df['pipeline'] == pipeline.name]["threshold"])

    print(f'{pipeline.name} - thresh - {thresh}')

    for value in tqdm.tqdm(data, total=len(data)):
        name = value[0].split('/')[-2]

        for file in tqdm.tqdm(value, total=len(value)):
            # print(f'{pipeline_name} - file - {file}')
            avg_emb = np.array(embedings[name])
            
            _st = timeit.default_timer()
            similarity, distance = pipeline(avg_emb, file)
            elapsed_time.append(timeit.default_timer() - _st)
            
            label = get_label(name, file)

            scores.append(similarity)
            labels.append(label)

    
    predictions = [True if score >= thresh else False for score in scores]

    percentage = int((sum(predictions) / len(predictions)) * 100)
    
    return percentage


def main(dataset_path, embedings_path, score_path):

    # Intended for experiment reproducibility
    if os.path.exists(f"{dataset_path}.pkl"):
        print(f"Loading {dataset_path}.pkl")
        with open(f"{dataset_path}.pkl", "rb") as f:
            dataset = pkl.load(f)
    else:
        print("Generating dataset")
        dataset  = make_dataset_3_4(f"{dataset_path}/")
        with open(f"{dataset_path}.pkl", "wb") as f:
            pkl.dump(dataset, f, protocol=pkl.HIGHEST_PROTOCOL)
    
    print(f"Number of synthesizers in dataset: {len(dataset)}")
    print(f"Number of speakers in dataset: {len(dataset[random.choice(list(dataset.keys()))])}")
    print(f"Number of samples per speaker in dataset: {len(random.choice(dataset[random.choice(list(dataset.keys()))]))}")

    # Define pipelines
    pipelines = [
        Pipeline("pyannote", Pyannote(), f"{embedings_path}/"),
        Pipeline("wavlm-base-sv", WavLM("microsoft/wavlm-base-sv", device="cpu"), f"{embedings_path}/"),
        Pipeline("wavlm-base-plus-sv", WavLM("microsoft/wavlm-base-plus-sv", device="cpu"), f"{embedings_path}/"),
        Pipeline("titanet", TitaNet(), f"{embedings_path}/"),
        Pipeline("ecapa", Ecapa(), f"{embedings_path}/"),
    ]

    results = {}

    for synthesizer, data in tqdm.tqdm(dataset.items()):
        print('Synthesizer - ', synthesizer)
        pipe_result = {}
        for pipeline in pipelines:
            print(f"Evaluating pipeline: {pipeline.name}")
            pipe_result[pipeline.name] = evaluate_pipeline(pipeline, data)

        results[synthesizer] = pipe_result
    
    # Store resutls in a csv file
    pd.DataFrame(results).transpose().to_csv(f"{score_path}.csv")


if __name__ == "__main__":

    main("dataset_exp_3", "embedings_exp_3", "scores_exp_3")

    print('EXPERIMENT 3 DONE')

    main("dataset_exp_4", "embedings_exp_4", "scores_exp_4")
