import nemo.collections.asr as nemo_asr

from core.audio import Audio


class TitaNet:

    def __init__(self):
        self.model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")
    
    def __call__(self, audio: Audio):
        return self.model.get_embedding(audio.path)


if __name__ == "__main__":
    model = TitaNet()

    audio1 = Audio("dataset/1-Zelenskyi/audio09.wav") 
    emb1 = model(audio1)

    print("Shape:", emb1.shape)
    print("Min:", emb1.min())
    print("Max:", emb1.max())
    print("AVG:", emb1.mean())
    print("STD:", emb1.std())
