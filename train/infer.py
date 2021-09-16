from model import CNN
import numpy as np
import torch

from transformers import ZmuvTransform, audio_transform
from pathlib import Path

import torch.nn.functional as F


import pyaudio

wake_words = ["hey", "fourth", "brain"]

sr = 16000

num_labels = len(wake_words) + 1  # oov
num_maps1 = 48
num_maps2 = 64
num_hidden_input = 768
hidden_size = 128
model = CNN(num_labels, num_maps1, num_maps2, num_hidden_input, hidden_size)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


model.load_state_dict(torch.load("model_hey_fourth_brain.pt"))

zmuv_transform = ZmuvTransform().to(device)
if Path("zmuv.pt.bin").exists():
    zmuv_transform.load_state_dict(torch.load(str("zmuv.pt.bin")))

classes = wake_words[:]
# oov
classes.append("oov")

audio_float_size = 32767
p = pyaudio.PyAudio()

CHUNK = 500
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = sr
RECORD_MILLI_SECONDS = 750

stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

print("* listening .. ")

inference_track = []
target_state = 0

while True:
    no_of_frames = 4

    # import pdb;pdb.set_trace()
    batch = []
    for frame in range(no_of_frames):
        frames = []
        for i in range(0, int(RATE / CHUNK * RECORD_MILLI_SECONDS / 1000)):
            data = stream.read(CHUNK)
            frames.append(data)
        audio_data = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float) / audio_float_size
        inp = torch.from_numpy(audio_data).float().to(device)
        batch.append(inp)

    audio_tensors = torch.stack(batch)

    # sav_temp_wav(frames)
    mel_audio_data = audio_transform(audio_tensors)
    mel_audio_data = zmuv_transform(mel_audio_data)
    scores = model(mel_audio_data.unsqueeze(1))
    scores = F.softmax(scores, -1).squeeze(1)  # [no_of_frames x num_labels]
    # import pdb;pdb.set_trace()
    for score in scores:
        preds = score.cpu().detach().numpy()
        preds = preds / preds.sum()
        # print([f"{x:.3f}" for x in preds.tolist()])
        pred_idx = np.argmax(preds)
        pred_word = classes[pred_idx]
        # print(f"predicted label {pred_idx} - {pred_word}")
        label = wake_words[target_state]
        if pred_word == label:
            target_state += 1  # go to next label
            inference_track.append(pred_word)
            print(inference_track)
            if inference_track == wake_words:
                print(f"Wake word {' '.join(inference_track)} detected")
                target_state = 0
                inference_track = []
                break
