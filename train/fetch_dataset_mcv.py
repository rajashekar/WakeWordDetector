import re
import math
from pathlib import Path

import pandas as pd
import soundfile
import librosa

# Dynamically get these paths from args
common_voice_datapath = "D:\\GoogleDrive\\datasets\\cv-corpus-6.1-2020-12-11\\en"
wake_words = ["hey", "fourth", "brain"]
wake_words_sequence = ["0", "1", "2"]

wake_word_datapath = "wake_word_ds"
positive_data = "/positive/audio"
negative_data = "/negative/audio"

sr = 16000

wake_word_seq_map = dict(zip(wake_words, wake_words_sequence))
regex_pattern = r"\b(?:{})\b".format("|".join(map(re.escape, wake_words)))
pattern = re.compile(regex_pattern, flags=re.IGNORECASE)


def wake_words_search(pattern, word):
    try:
        return bool(pattern.search(word))
    except TypeError:
        return False


def save_wav_lab(path, filename, sentence, decibels=40):
    # load file
    sounddata = librosa.core.load(f"{common_voice_datapath}/clips/{filename}", sr=sr, mono=True)[0]
    # trim
    sounddata = librosa.effects.trim(sounddata, top_db=decibels)[0]
    # save as wav file
    soundfile.write(f"{wake_word_datapath}{path}/{filename.split('.')[0]}.wav", sounddata, sr)
    # write lab file
    with open(f"{wake_word_datapath}{path}/{filename.split('.')[0]}.lab", "w", encoding="utf-8") as f:
        f.write(sentence)


train_data = pd.read_csv("train.tsv", sep="\t")
dev_data = pd.read_csv("dev.tsv", sep="\t")
test_data = pd.read_csv("test.tsv", sep="\t")

positive_train_data = train_data[[wake_words_search(pattern, sentence) for sentence in train_data["sentence"]]]
positive_dev_data = dev_data[[wake_words_search(pattern, sentence) for sentence in dev_data["sentence"]]]
positive_test_data = test_data[[wake_words_search(pattern, sentence) for sentence in test_data["sentence"]]]

negative_train_data = train_data[[not wake_words_search(pattern, sentence) for sentence in train_data["sentence"]]]
negative_dev_data = dev_data[[not wake_words_search(pattern, sentence) for sentence in dev_data["sentence"]]]
negative_test_data = test_data[[not wake_words_search(pattern, sentence) for sentence in test_data["sentence"]]]

print(f"Total clips available in Train {train_data.shape[0]}")
print(f"Total clips available in Dev {dev_data.shape[0]}")
print(f"Total clips available in Test {test_data.shape[0]}")

print(f"Total clips available in Train with wake words {positive_train_data.shape[0]}")
print(f"Total clips available in Dev with wake words {positive_dev_data.shape[0]}")
print(f"Total clips available in Test with wake words {positive_test_data.shape[0]}")

# negative data size
print(f"Total clips available in Train without wake words {negative_train_data.shape[0]}")
print(f"Total clips available in Dev without wake words {negative_dev_data.shape[0]}")
print(f"Total clips available in Test without wake words {negative_test_data.shape[0]}")

# trim negative data size to 1%
negative_data_percent = 1
negative_train_data = negative_train_data.sample(
    math.floor(negative_train_data.shape[0] * (negative_data_percent / 100))
)
negative_dev_data = negative_dev_data.sample(math.floor(negative_dev_data.shape[0] * (negative_data_percent / 100)))
negative_test_data = negative_test_data.sample(math.floor(negative_test_data.shape[0] * (negative_data_percent / 100)))

# trimmed negative data sizes
print(f"Total clips available in Train without wake words {negative_train_data.shape[0]}")
print(f"Total clips available in Dev without wake words {negative_dev_data.shape[0]}")
print(f"Total clips available in Test without wake words {negative_test_data.shape[0]}")


Path(wake_word_datapath).mkdir(parents=True, exist_ok=True)
# create postiive & negative dataset folder
Path(wake_word_datapath + positive_data).mkdir(parents=True, exist_ok=True)
Path(wake_word_datapath + negative_data).mkdir(parents=True, exist_ok=True)

# save the dataframes we got from above in each dataset
positive_train_data[["path", "sentence"]].to_csv(wake_word_datapath + "/positive/train.csv", index=False)
positive_dev_data[["path", "sentence"]].to_csv(wake_word_datapath + "/positive/dev.csv", index=False)
positive_test_data[["path", "sentence"]].to_csv(wake_word_datapath + "/positive/test.csv", index=False)

negative_train_data[["path", "sentence"]].to_csv(wake_word_datapath + "/negative/train.csv", index=False)
negative_dev_data[["path", "sentence"]].to_csv(wake_word_datapath + "/negative/dev.csv", index=False)
negative_test_data[["path", "sentence"]].to_csv(wake_word_datapath + "/negative/test.csv", index=False)


positive_train_data.progress_apply(lambda x: save_wav_lab(positive_data, x["path"], x["sentence"]), axis=1)
positive_dev_data.progress_apply(lambda x: save_wav_lab(positive_data, x["path"], x["sentence"]), axis=1)
positive_test_data.progress_apply(lambda x: save_wav_lab(positive_data, x["path"], x["sentence"]), axis=1)

negative_train_data.progress_apply(lambda x: save_wav_lab(negative_data, x["path"], x["sentence"]), axis=1)
negative_dev_data.progress_apply(lambda x: save_wav_lab(negative_data, x["path"], x["sentence"]), axis=1)
negative_test_data.progress_apply(lambda x: save_wav_lab(negative_data, x["path"], x["sentence"]), axis=1)
