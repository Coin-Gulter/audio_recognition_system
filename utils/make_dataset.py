import os
import glob
import random
from typing import List, Tuple

from utils import generate_speakers_file_pairs


def make_dataset(dataset_dir: str, without_pairs=False) -> List[Tuple[str, str]]:
    # Get speaker to samples map
    speakers = glob.glob(os.path.join(dataset_dir, "*"))
    speaker_to_samples_dict = { s: glob.glob(os.path.join(s, "*.wav")) for s in speakers }
    # Generate datasets for evaluation for each speaker
    print('\n\nspeakers - ' ,speakers)
    print('\n\nspeakers_to_sample - ' ,speaker_to_samples_dict)

    speakers_samples_list = []

    if without_pairs:
        for s in speakers:
            speaker_temp_list = speaker_to_samples_dict[s]
            
            i = 0
            while i <= len(speakers):
                speaker = random.choice(speakers)

                if speaker == s:
                    continue
                else:
                    index = random.randint(0,len(speaker_to_samples_dict[speaker])-1)
                    speaker_temp_list.append(speaker_to_samples_dict[speaker][index])
                    i += 1

            speakers_samples_list.append(tuple(speaker_temp_list))

        print('without making pairs')
        return speakers_samples_list
    
    dataset_dict, dataset_list  = generate_speakers_file_pairs(speaker_to_samples_dict)

    return dataset_list
