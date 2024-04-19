import os
import tqdm
import numpy as np
import glob
import json
from core.pyannote import Audio

from core import (
    Pyannote,
    WavLM,
    TitaNet,
    Ecapa
)

from core.audio import Audio
from core.normalize import normalize




def avg_embedding(embeddings):
  """
  This function takes a list of embeddings and returns the average embedding.

  Args:
      embeddings: A list of numpy arrays, where each array represents an embedding.

  Returns:
      A numpy array representing the average embedding.
  """


  # Get the average embedding by dividing the sum by the number of embeddings
  avg_embedding = np.mean(embeddings, axis=0)


  return avg_embedding

def get_pipeline(name: str):
    if name == "ecapa":
        return Ecapa()
    elif name == "titanet":
        return TitaNet()
    


def main(dataset_dir = './dataset/'):
    speakers = glob.glob(os.path.join(dataset_dir, "*"))
    speaker_to_samples_dict = { s: glob.glob(os.path.join(s, "*.wav")) for s in speakers }
    # Generate datasets for evaluation for each speaker
    print('\n\nspeakers - ' ,speakers)
    print('\n\nspeakers_to_sample - ' ,speaker_to_samples_dict)

    # Define pipelines
    emb_functions = {
        "pyannote": Pyannote(),
        "wavlm-base-sv": WavLM("microsoft/wavlm-base-sv", device="cpu"),
        "wavlm-base-plus-sv": WavLM("microsoft/wavlm-base-plus-sv", device="cpu"),
        "titanet": TitaNet(),
        "ecapa": Ecapa(),
    }

    for function in tqdm.tqdm(emb_functions, total=len(emb_functions)):
        speakers_embeddings_dict = {}
        speaker_avg_dict = {}
        for speaker, values in tqdm.tqdm(speaker_to_samples_dict.items()):
            speaker = speaker.split('/')[-1]

            speaker_embeddings_list = np.array([])
            speaker_dict = {}

            for v in values:

                if isinstance(v, str):
                    audio = Audio(v)
                else:
                    audio = v

                embeding = emb_functions[function](audio)
                embeding = normalize(embeding)

                if not isinstance(embeding, np.ndarray):
                    embeding = embeding.detach().numpy()

                if len(speaker_embeddings_list) == 0:
                    speaker_embeddings_list = np.expand_dims(embeding, axis=0)
                else:
                    speaker_embeddings_list = np.append(speaker_embeddings_list, np.expand_dims(embeding, axis=0), axis=0)

                speaker_dict[v] = embeding.tolist()

            avg_embedding = np.mean(speaker_embeddings_list, axis=0)

            speaker_avg_dict[speaker] = avg_embedding.tolist()
            speakers_embeddings_dict[speaker] = speaker_dict

        with open(f"embedings_exp_1/{function}.json", "+w") as f:
            f.write(json.dumps(speakers_embeddings_dict))

        with open(f"avg_embedings/{function}.json", "+w") as f:
            f.write(json.dumps(speaker_avg_dict))

if __name__=="__main__":
    main(dataset_dir='dataset_exp_1/')