import time
from pathlib import Path
import os

import numpy as np
import pandas as pd

from google.cloud import texttospeech

# Instantiates a client
client = texttospeech.TextToSpeechClient()

wake_word_datapath = "wake_word_ds"
generated_data = "generated"
wake_words = ["hey", "fourth", "brain"]

Path(f"{wake_word_datapath}/{generated_data}").mkdir(parents=True, exist_ok=True)


def generate_voices(word):
    Path(f"{wake_word_datapath}/{generated_data}/{word}").mkdir(parents=True, exist_ok=True)
    # Set the text input to be synthesized
    synthesis_input = texttospeech.SynthesisInput(text=word)
    # Performs the list voices request
    voices = client.list_voices()
    # Get english voices
    en_voices = [voice.name for voice in voices.voices if voice.name.split("-")[0] == "en"]
    speaking_rates = np.arange(0.25, 4.25, 0.25).tolist()
    pitches = np.arange(-10.0, 10.0, 2).tolist()
    file_count = 0
    start = time.time()

    for voi in en_voices:
        for sp_rate in speaking_rates:
            for pit in pitches:
                file_name = f"{wake_word_datapath}/{generated_data}/{word}/{voi}_{sp_rate}_{pit}.wav"
                voice = texttospeech.VoiceSelectionParams(language_code=voi[:5], name=voi)
                # Select the type of audio file you want returned
                audio_config = texttospeech.AudioConfig(
                    # format of the audio byte stream.
                    audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                    # Speaking rate/speed, in the range [0.25, 4.0]. 1.0 is the normal native speed
                    speaking_rate=sp_rate,
                    # Speaking pitch, in the range [-20.0, 20.0]. 20 means increase 20 semitones from
                    # the original pitch. -20 means decrease 20 semitones from the original pitch.
                    pitch=pit,  # [-10, -5, 0, 5, 10]
                )
                response = client.synthesize_speech(
                    request={"input": synthesis_input, "voice": voice, "audio_config": audio_config}
                )
                # The response's audio_content is binary.
                with open(file_name, "wb") as out:
                    out.write(response.audio_content)
                    file_count += 1
                if file_count % 100 == 0:
                    end = time.time()
                    print(f"generated {file_count} files in {end-start} seconds")


# generate for each word
for word in wake_words:
    generate_voices(word)

for word in wake_words:
    d = {}
    d["path"] = [
        f"{generated_data}/{word}/{file_name}"
        for file_name in os.listdir(f"{wake_word_datapath}/{generated_data}/{word}")
    ]
    d["sentence"] = [word] * len(d["path"])
    pd.DataFrame(data=d).to_csv(f"{generated_data}/{word}.csv", index=False)


word_cols = {"path": [], "sentence": []}
train, dev, test = pd.DataFrame(word_cols), pd.DataFrame(word_cols), pd.DataFrame(word_cols)
for word in wake_words:
    word_df = pd.read_csv(f"{generated_data}/{word}.csv")
    tra, val, te = np.split(word_df.sample(frac=1, random_state=42), [int(0.6 * len(word_df)), int(0.8 * len(word_df))])
    train = pd.concat([train, tra]).sample(frac=1).reset_index(drop=True)
    dev = pd.concat([dev, val]).sample(frac=1).reset_index(drop=True)
    test = pd.concat([test, te]).sample(frac=1).reset_index(drop=True)

train.to_csv(f"{generated_data}/train.csv", index=False)
dev.to_csv(f"{generated_data}/dev.csv", index=False)
test.to_csv(f"{generated_data}/test.csv", index=False)
