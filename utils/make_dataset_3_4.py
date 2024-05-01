import os
import glob
import random
from typing import List, Tuple



def make_dataset_3_4(dataset_dir: str):

    synthesizers = glob.glob(os.path.join(dataset_dir, "*"))

    synthesizers_sample_dict = {}

    for syntesizer in synthesizers:
        # Get speaker to samples map
        speakers = glob.glob(os.path.join(syntesizer, "*"))
        speaker_to_samples_dict = { s: glob.glob(os.path.join(s, "*")) for s in speakers }
        # Generate datasets for evaluation for each speaker
        print('\n\nspeakers - ' ,speakers)
        print('\n\nspeakers_to_sample - ' ,speaker_to_samples_dict)

        speakers_samples_list = []

        for s in speakers:
            speaker_temp_list = speaker_to_samples_dict[s]

            speakers_samples_list.append(tuple(speaker_temp_list))
        synthesizers_sample_dict[syntesizer] = speakers_samples_list

    print('dataset generation done')
    return synthesizers_sample_dict
