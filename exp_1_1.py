import os
import tqdm
import timeit
import numpy as np
import pandas as pd
import pickle as pkl
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

from utils import make_dataset


def get_pipeline(name: str):
    if name == "ecapa":
        return Ecapa()
    elif name == "titanet":
        return TitaNet()



def get_label(file1: str, file2: str) -> int:
    """
    Return 0 if different speakers, 1 if same speakers.
    """
    def _get_name(x):
        return x.split("/")[-2]

    return int(_get_name(file1) == _get_name(file2))


def evaluate_pipeline(
    pipeline, 
    data, 
) -> pd.DataFrame:

    scores = []
    labels = []
    distance_self = []
    distance_other = []
    elapsed_time = []

    for file1, file2 in tqdm.tqdm(data, total=len(data)):

        _st = timeit.default_timer()
        similarity, distance = pipeline(file1, file2)
        elapsed_time.append(timeit.default_timer() - _st)
        
        label = get_label(file1, file2)
        
        scores.append(similarity)
        labels.append(label)
        
        if label == 1:
            distance_self.append(distance)
        else:
            distance_other.append(distance)


    ee_rate, thresh, fa_rate, fr_rate = compute_eer(scores, labels)
    min_dcf = compute_min_dcf(fr_rate, fa_rate)
    fa_score, fr_score = compute_far_frr(scores, labels, thresh)
    
    predictions = [1 if score >= thresh else 0 for score in scores]

    dist_self_mean, dist_self_std = np.mean(distance_self), np.std(distance_self)
    dist_other_mean, dist_other_std = np.mean(distance_other), np.std(distance_other)
    elapsed_time_mean, elapsed_time_std = np.mean(elapsed_time), np.std(elapsed_time)
    
    result = {
        "pipeline": pipeline.name,
        "fa_score": fa_score,
        "fr_score": fr_score,
        "ee_rate": ee_rate,
        "dcf": min_dcf, 
        "threshold": thresh,
        "accuracy": accuracy_score(labels, predictions),
        "distance_self_mean": dist_self_mean,
        "distance_self_std": dist_self_std,
        "distance_other_mean": dist_other_mean,
        "distance_other_std": dist_other_std,
        "elapsed_time_mean": elapsed_time_mean,
        "elapsed_time_std": elapsed_time_std,
    }
    return result


def main():

    # Intended for experiment reproducibility
    if os.path.exists("dataset_exp_1.pkl"):
        print("Loading dataset_exp_1.pkl")
        with open("dataset_exp_1.pkl", "rb") as f:
            dataset = pkl.load(f)
    else:
        print("Generating dataset")
        dataset  = make_dataset("dataset_exp_1/")
        print("Saving dataset_exp_1.pkl")
        with open("dataset_exp_1.pkl", "wb") as f:
            pkl.dump(dataset, f)
    
    print(f"Number of pairs in dataset: {len(dataset)}")

    # Define pipelines
    pipelines = [
        Pipeline("pyannote", Pyannote(), "embedings_exp_1/"),
        Pipeline("wavlm-base-sv", WavLM("microsoft/wavlm-base-sv", device="cpu"), "embedings_exp_1/"),
        Pipeline("wavlm-base-plus-sv", WavLM("microsoft/wavlm-base-plus-sv", device="cpu"), "embedings_exp_1/"),
        Pipeline("titanet", TitaNet(), "embedings_exp_1/"),
        Pipeline("ecapa", Ecapa(), "embedings_exp_1/"),
    ]

    results = {}

    for pipeline in pipelines:
        print(f"Evaluating pipeline: {pipeline.name}")
        results[pipeline.name] = evaluate_pipeline(pipeline, dataset)
    
    # Store resutls in a csv file
    pd.DataFrame(results).transpose().to_csv("scores_exp_1.csv")

    # # Make visualizations
    # speakers = [
    #     '1-Zelenskyi',
    #     '2-Sadovyi',
    #     '9-Vereshchuk',
    #     '10-Kuleba',
    #     '25-Ustinova',
    # ]
    # markers = [
    #     'o',
    #     'x',
    #     's',
    #     '+',
    #     '^',
    # ]
    # colors = [
    #     'r',
    #     'g',
    #     'b',
    #     'm',
    #     'c',
    # ]
    # make_visualization(pipelines, dataset, speakers=speakers, markers=markers, colors=colors)
        
    print("Done!")



# def make_visualization(
#     pipelines,
#     dataset,
#     speakers=None,
#     markers=None,
#     colors=None,
# ):
#     classes = {}
#     for file1, file2 in dataset:
#         if get_label(file1, file2) == 1:
#             name = file1.split("/")[-2]
#             if name not in classes.keys():
#                 classes[name] = [file1]
#             else:
#                 classes[name].append(file1)
    
#     # Limit number of classes
#     if speakers is None:
#         selected_classes_str = [f"{i}-" for i in range(1, 11)]
#     else:
#         selected_classes_str = speakers
#     classes = {k: v for k, v in classes.items() if any([k.startswith(s) for s in selected_classes_str])}
#     speaker_to_marker_map = dict(zip(speakers, markers))
#     speaker_to_color_map = dict(zip(speakers, colors))
    
#     fig, axs = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(8, 12))
#     fig.delaxes(axs[2, 1])
    
#     for i, pipeline in enumerate(pipelines):
        
#         ax_x, ax_y = i // 2, i % 2
#         ax = axs[ax_x, ax_y]
        
#         class_embeddings = {}
        
#         for speaker, files in tqdm.tqdm(classes.items(), desc=f"Visualising {pipeline.name}"):
            
#             class_embeddings[speaker] = []
            
#             for f in files:
#                 emb = normalize(pipeline.embedding_fn(Audio(f)))
#                 class_embeddings[speaker].append(emb)
            
#         all_embeddings = [emb for embs in class_embeddings.values() for emb in embs]

#         ax.set_title(pipeline.name)
#         ax.set_yticklabels([])
#         ax.set_xticklabels([])
    
#         pca = PCA(n_components=2).fit(all_embeddings)
        
#         for speaker, embs in class_embeddings.items():
#             decomp_embs = pca.transform(embs)
#             _x = decomp_embs[:, 0]
#             _y = decomp_embs[:, 1]
#             _m = speaker_to_marker_map[speaker]
#             _c = speaker_to_color_map[speaker]
#             ax.scatter(_x, _y, c=_c, marker=_m, label=speaker)
        
#         handles, labels = ax.get_legend_handles_labels()
        
#         extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
#         fig.savefig(f'plots/{pipeline.name}.png', bbox_inches=extent.expanded(1.2, 1.2))
        
#     os.makedirs("plots", exist_ok=True) 
#     fig.legend(handles, labels, ncol=len(speakers), loc='lower center')
#     plt.savefig(f"plots/total.png")


if __name__ == "__main__":
    main()
