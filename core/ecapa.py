import torchaudio
from speechbrain.inference.classifiers import EncoderClassifier

from core.audio import Audio


class Ecapa:

    def __init__(self):
        self.model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device":"cpu"})
    
    def __call__(self, audio: Audio):
        signal, fs = torchaudio.load(audio.path)
        return self.model.encode_batch(signal)


if __name__ == "__main__":
    model = Ecapa()

    audio1 = Audio("dataset/1-Zelenskyi/audio09.wav") 
    emb1 = model(audio1)

    print("Shape:", emb1.shape)
    print("Min:", emb1.min())
    print("Max:", emb1.max())
    print("AVG:", emb1.mean())
    print("STD:", emb1.std())
