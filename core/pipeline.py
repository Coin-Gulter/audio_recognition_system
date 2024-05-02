from typing import Callable, Tuple

import numpy as np
import json
import os

from core.pyannote import Audio
from core.cosine_similarity import cosine_similarity
from core.distance import euclidean_distance
from core.normalize import normalize


class Pipeline:

    def __init__(
            self, 
            name: str,
            embedding_fn: Callable,
            cache_dir: str
            # score_fn: Callable,
        ):
        self.name = name
        self.embedding_fn = embedding_fn
        self.cache_dir = cache_dir
        # self.score_fn = score_fn

    def __call__(
            self, 
            audio1: Audio | str | np.ndarray, 
            audio2: Audio | str | np.ndarray
        ) -> Tuple[float, float]:

        if isinstance(audio1, np.ndarray):
            emb1 = audio1
        elif isinstance(audio1, str):
            check_emb1 = self.check_if_emb_exist(audio1)

            if isinstance(check_emb1, np.ndarray):
                emb1 = check_emb1
            else:
                audio1 = Audio(audio1)
                emb1 = self.embedding_fn(audio1)
                emb1 = normalize(emb1)
                self.cache_emb(audio1, emb1)

        if isinstance(audio2, np.ndarray):
            emb2 = audio2
        elif isinstance(audio2, str):
            check_emb2 = self.check_if_emb_exist(audio2)

            if isinstance(check_emb2, np.ndarray):
                emb2 = check_emb2
            else:
                audio_file2 = audio2
                audio2 = Audio(audio2)
                emb2 = self.embedding_fn(audio2)
                emb2 = normalize(emb2)
                self.cache_emb(audio_file2, emb2)
        
        # similarity = self.score_fn(emb1, emb2)
        similarity = cosine_similarity(emb1, emb2)
        distance = euclidean_distance(emb1, emb2)
        return (similarity, distance)
    
    def check_if_emb_exist(self, audio: str) -> (np.ndarray | bool):

        file_path = f"{self.cache_dir + self.name}.json"

        if os.path.isfile(file_path):

            with open(file_path, "+r") as f:
                speakers_emd_dict = json.load(f)

            speaker = audio.split("/")[-2]

            if speaker in speakers_emd_dict:
                speaker_files_emd_dict = speakers_emd_dict[speaker]

                if audio in speaker_files_emd_dict:
                    return np.array(speaker_files_emd_dict[audio])
                
                else:
                    False
            else:
                False
        else:
            False

    def cache_emb(self, file, emb):
        file_path = f"{self.cache_dir + self.name}.json"
        speakers_emd_dict = {}

        if os.path.isfile(file_path):
            with open(file_path, "+r") as f:
                speakers_emd_dict = json.load(f)

        speaker = file.split("/")[-2]

        if speaker in speakers_emd_dict:
            speaker_embedings_dict = speakers_emd_dict[speaker]
            speaker_embedings_dict[file] = emb.tolist()
        else:
            speaker_embedings_dict = {}
            speaker_embedings_dict[file] = emb.tolist()
            
        speakers_emd_dict[speaker] = speaker_embedings_dict

        os.makedirs(self.cache_dir, exist_ok=True)
        with open(file_path, "+w") as f:
            f.write(json.dumps(speakers_emd_dict))        


if __name__ == "__main__":
    from core.pyannote import Pyannote, cosine_similarity

    audio1 = Audio("dataset/1-Zelenskyi/audio01.wav")
    audio2 = Audio("dataset/2-Sadovyi/audio01.wav")

    pipeline = Pipeline("pyannote", Pyannote())
    print(pipeline(audio1, audio2))
