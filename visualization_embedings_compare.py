import os
import matplotlib.pyplot as plt
import tqdm
import json
from core.normalize import normalize
from sklearn.decomposition import PCA


get_label = lambda name, index: name.split("/")[-(index)]


def make_visualization(
    pipelines,
    dataset,
    speakers,
    markers,
    colors,
):
    dataset_to_marker_map = dict(zip(dataset, markers))
    speaker_to_color_map = dict(zip(speakers, colors))
    
    fig, axs = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(15, 12))
    fig.delaxes(axs[2, 1])
    
    for dt in dataset:
        for i, pipeline_name in enumerate(pipelines):
            classes = {}

            with open(f"{dt}/{pipeline_name}.json") as f:
                classes_json = json.load(f)

            for class_name, embeding in classes_json.items():
                if class_name in speakers:
                    embedings = []
                    for file_name, emb in embeding.items():
                        embedings.append(emb)

                    if len(embedings) > 10:
                        embedings = embedings[:10]
                    classes[f"{dt}/{class_name}"] = embedings

            
            ax_x, ax_y = i // 2, i % 2
            ax = axs[ax_x, ax_y]
            
            class_embeddings = {}
            
            for speaker, embedings in tqdm.tqdm(classes.items(), desc=f"Visualising {pipeline_name}"):
                
                class_embeddings[speaker] = []
                
                for emb in embedings:
                    emb = normalize(emb)
                    class_embeddings[speaker].append(emb)
                
            all_embeddings = [emb for embs in class_embeddings.values() for emb in embs]

            ax.set_title(pipeline_name)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
        
            pca = PCA(n_components=2).fit(all_embeddings)
            
            for speaker, embs in class_embeddings.items():
                decomp_embs = pca.transform(embs)
                _x = decomp_embs[:, 0]
                _y = decomp_embs[:, 1]
                _m = dataset_to_marker_map[get_label(speaker, 2)]
                _c = speaker_to_color_map[get_label(speaker, 1)]
                ax.scatter(_x, _y, c=_c, marker=_m, label=speaker)
            
            handles, labels = ax.get_legend_handles_labels()
            
            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(f'plots/{pipeline_name}.png', bbox_inches=extent.expanded(1.2, 1.2))
        
    os.makedirs("plots", exist_ok=True) 
    fig.legend(handles, labels, ncol=len(speakers), loc='lower center')
    plt.savefig(f"plots/total.png")

def main():
    # Make visualizations
    pipelines = [
        "pyannote",
        "ecapa",
        "titanet",
        "wavlm-base-plus-sv",
        "wavlm-base-sv"
    ]
    dataset = [
        "voices_original",
        "voices_cloning"
    ]
    speakers = [
        'AlanRickman',
        'AnnaMassey',
        'BenedictCumberbatch',
        'NeilGaiman',
        'TomHanks',
    ]
    ds_markers = [
        'o',
        'x'
        # 's',
        # '+',
        # '^',
    ]
    colors = [
        'r',
        'g',
        'b',
        'm',
        'c',
    ]
    make_visualization(pipelines=pipelines, dataset=dataset, speakers=speakers, markers=ds_markers, colors=colors)


if __name__ == "__main__":
    main()