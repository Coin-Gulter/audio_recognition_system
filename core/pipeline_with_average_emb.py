from typing import Callable, Tuple

import torch
import numpy as np

from core.pyannote import Audio
from core.cosine_similarity import cosine_similarity
from core.distance import euclidean_distance
from core.normalize import normalize


class Pipeline_avg:

    def __init__(
            self, 
            name: str,
            embedding_fn: Callable, 
            # score_fn: Callable,
        ):
        self.name = name
        self.embedding_fn = embedding_fn
        # self.score_fn = score_fn

    def __call__(
            self, 
            enmbeding, 
            audio: Audio or str
        ) -> Tuple[float, float]:

        if isinstance(audio, str):
            audio = Audio(audio)

        emb = self.embedding_fn(audio)
        emb = normalize(emb)
        
        # similarity = self.score_fn(emb1, emb2)
        similarity = cosine_similarity(enmbeding, emb)
        distance = euclidean_distance(enmbeding, emb)
        return (similarity, distance)
