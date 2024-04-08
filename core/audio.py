import soundfile as sf
import librosa as ls


class Audio:

    def __init__(
        self, 
        path: str, 
        sample_rate: int = 16000, 
        mono: bool = True,
        overwrite: bool = True,
    ) -> None:
        _wav, _sr = ls.load(path, sr=sample_rate, mono=mono)
        self.path = path
        self.wave = _wav
        self.sample_rate = _sr

        if overwrite:
            sf.write(path, _wav, _sr)
    
