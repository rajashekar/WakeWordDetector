import random
import re
import json

import librosa
import numpy as np
import torch

sr = 16000

key_pattern = re.compile("'(?P<k>[^ ]+)'")


def compute_labels(metadata, audio_data, wake_words, wake_word_seq_map):
    label = len(wake_words)  # by default negative label

    # if it is generated data then
    if metadata["sentence"].lower() in wake_words:
        label = int(wake_word_seq_map[metadata["sentence"].lower()])
    else:
        # if the sentence has one wakeword get label and end timestamp
        for word in metadata["sentence"].lower().split():
            wake_word_found = False
            word = re.sub("\W+", "", word)
            if word in wake_words:
                wake_word_found = True
                break

        if wake_word_found:
            label = int(wake_word_seq_map[word])
            if word in metadata["timestamps"]:
                timestamps = metadata["timestamps"]
                if type(timestamps) == str:
                    timestamps = json.loads(key_pattern.sub(r'"\g<k>"', timestamps))
                word_ts = timestamps[word]
                audio_start_idx = int((word_ts["start"] * 1000) * sr / 1000)
                audio_end_idx = int((word_ts["end"] * 1000) * sr / 1000)
                audio_data = audio_data[audio_start_idx:audio_end_idx]
            else:  # if there are issues with word alignment, we might not get ts
                label = len(wake_words)  # mark them for negative

    return label, audio_data


class AudioCollator(object):
    def __init__(self, wake_words, wake_word_seq_map, noise_set=None):
        self.noise_set = noise_set
        self.wake_words = wake_words
        self.wake_word_seq_map = wake_word_seq_map

    def __call__(self, batch):
        batch_tensor = {}
        window_size_ms = 750
        max_length = int(window_size_ms / 1000 * sr)
        audio_tensors = []
        labels = []
        for sample in batch:
            # get audio_data in tensor format
            audio_data = librosa.core.load(sample["path"], sr=sr, mono=True)[0]
            # get the label and its audio
            label, audio_data = compute_labels(sample, audio_data, self.wake_words, self.wake_word_seq_map)
            audio_data_length = audio_data.size / sr * 1000  # ms

            # below is to make sure that we always got length of 12000
            # i.e 750 ms with sr 16000
            # trim to max_length
            if audio_data_length > window_size_ms:
                # randomly trim either at start and end
                if random.random() < 0.5:
                    audio_data = audio_data[:max_length]
                else:
                    audio_data = audio_data[audio_data.size - max_length :]  # noqa

            # pad with zeros
            if audio_data_length < window_size_ms:
                # randomly either append or prepend
                if random.random() < 0.5:
                    audio_data = np.append(audio_data, np.zeros(int(max_length - audio_data.size)))
                else:
                    audio_data = np.append(np.zeros(int(max_length - audio_data.size)), audio_data)

            # Add noise
            if self.noise_set:
                noise_level = random.randint(1, 5) / 10  # 10 to 50%
                noise_sample = librosa.core.load(
                    self.noise_set[random.randint(0, len(self.noise_set) - 1)], sr=sr, mono=True
                )[0]
                # randomly select first or last seq of noise
                if random.random() < 0.5:
                    audio_data = (1 - noise_level) * audio_data + noise_level * noise_sample[:max_length]
                else:
                    audio_data = (1 - noise_level) * audio_data + noise_level * noise_sample[-max_length:]

            audio_tensors.append(torch.from_numpy(audio_data))
            labels.append(label)

        batch_tensor = {"audio": torch.stack(audio_tensors), "labels": torch.tensor(labels)}

        return batch_tensor
