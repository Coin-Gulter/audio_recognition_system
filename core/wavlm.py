# from datasets import load_dataset
# import torch

# dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")

# feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-plus-sv')
# model = WavLMForXVector.from_pretrained('microsoft/wavlm-base-plus-sv')

# # audio files are decoded on the fly
# audio = [x["array"] for x in dataset[:2]["audio"]]
# inputs = feature_extractor(audio, padding=True, return_tensors="pt")
# embeddings = model(**inputs).embeddings
# embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()

# # the resulting embeddings can be used for cosine similarity-based retrieval
# cosine_sim = torch.nn.CosineSimilarity(dim=-1)
# similarity = cosine_sim(embeddings[0], embeddings[1])
# threshold = 0.86  # the optimal threshold is dataset-dependent
# if similarity < threshold:
#     print("Speakers are not the same!")

import torch
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector, WavLMForAudioFrameClassification

from core.audio import Audio



class WavLM:

    DEFAULT_MODEL_PATH = "microsoft/wavlm-base-plus-sv"

    def __init__(self, model_path: str = None, device: str = None):
        if model_path is None:
            model_path = self.DEFAULT_MODEL_PATH
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if device is not None:
            self.device = device
            
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        self.model = WavLMForXVector.from_pretrained(model_path).to(self.device)

    def __call__(self, audio: Audio):

        inputs = self.feature_extractor(
            [audio.wave], 
            sampling_rate=audio.sample_rate, 
            padding=True, 
            return_tensors="pt",
        )
        
        embeddings = self.model(**inputs.to(self.device)).embeddings

        return embeddings[0]


if __name__ == "__main__":
    model = WavLM()

    audio1 = Audio("dataset/1-Zelenskyi/audio09.wav") 
    emb1 = model(audio1)

    print("Shape:", emb1.shape)
    print("Min:", emb1.min())
    print("Max:", emb1.max())
    print("AVG:", emb1.mean())
    print("STD:", emb1.std())
