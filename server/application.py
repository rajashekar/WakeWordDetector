import json
import io
import base64
import threading


from flask import Flask, session, render_template, copy_current_request_context
from flask_socketio import emit, SocketIO

import numpy as np
import librosa

import matplotlib
import matplotlib.pyplot as plt

# import soundfile as sf
import torch
import torch.nn.functional as F

from utils.model import CNN
from utils.transformers import audio_transform, ZmuvTransform

matplotlib.use("Agg")
plt.ioff()

application = app = Flask(__name__)
app.config["FILEDIR"] = "static/_files/"

# socketio = SocketIO(app, logger=True, engineio_logger=True)
socketio = SocketIO(app, cors_allowed_origins="*")

wake_words = ["hey", "fourth", "brain"]
classes = wake_words[:]
classes.append("oov")

window_size_ms = 750
# 16 bit signed int. 2^15-1
audio_float_size = 32767
sample_rate = 16000
max_length = int(window_size_ms / 1000 * sample_rate)
no_of_frames = 2

# init model
num_labels = len(wake_words) + 1  # oov
num_maps1 = 48
num_maps2 = 64
num_hidden_input = 768
hidden_size = 128
model = CNN(num_labels, num_maps1, num_maps2, num_hidden_input, hidden_size)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
print(model)

# load trained model
model.load_state_dict(
    torch.load("trained_models/model_hey_fourth_brain_with_noise.pt", map_location=torch.device("cpu"))
)

# load zmuv
zmuv_transform = ZmuvTransform().to(device)
zmuv_transform.load_state_dict(torch.load(str("trained_models/zmuv.pt.bin"), map_location=torch.device("cpu")))


def plot_spectrogram(result, spec, title=None, ylabel="freq_bin", aspect="auto", xmax=None):
    mel_fig, mel_axs = plt.subplots(1, 1)
    mel_axs.set_title(title or "Spectrogram (db)")
    mel_axs.set_ylabel(ylabel)
    mel_axs.set_xlabel("frame")
    im = mel_axs.imshow(librosa.power_to_db(spec), origin="lower", aspect=aspect)
    if xmax:
        mel_axs.set_xlim((0, xmax))

    mel_fig.colorbar(im, ax=mel_axs)

    my_stringIObytes = io.BytesIO()
    plt.savefig(my_stringIObytes, format="jpg")
    my_stringIObytes.seek(0)
    my_base64_jpgData = base64.b64encode(my_stringIObytes.read())
    result["plot"] = my_base64_jpgData.decode("utf-8")


def plot_time(result, soundata):
    time_fig, time_axs = plt.subplots(1, 1)
    time_axs.set_title("Signal")
    time_axs.set_xlabel("Time (samples)")
    time_axs.set_ylabel("Amplitude")
    time_axs.plot(soundata)

    # plt.plot(soundata)
    my_stringIObytes = io.BytesIO()
    plt.savefig(my_stringIObytes, format="jpg")
    my_stringIObytes.seek(0)
    my_base64_jpgData = base64.b64encode(my_stringIObytes.read())
    result["plot"] = my_base64_jpgData.decode("utf-8")


@app.route("/")
def index():
    """Return the client application."""
    return render_template("audio/main.html")


def init_session(options=None):
    if options:
        session["bufferSize"] = options.get("bufferSize", 1024)
        session["fps"] = options.get("fps", 44100)
    else:
        session["bufferSize"] = 1024
        session["fps"] = 44100
    session["windowSize"] = 0
    session["frames"] = []
    session["batch"] = []
    session["target_state"] = 0
    session["infer_track"] = []
    session["threads"] = []


@socketio.on("start-recording", namespace="/audio")
def start_recording(options):
    """Start recording audio from the client."""
    init_session(options)


@socketio.on("write-audio", namespace="/audio")
def write_audio(data):

    """Write a chunk of audio from the client."""
    if "frames" not in session:
        init_session()
    session["frames"].append(data)
    session["windowSize"] += 1
    # print(f'{session["windowSize"]} - {int(session["fps"] / session["bufferSize"] * window_size_ms / 1000)}')
    # if we got 750 ms then process
    if session["windowSize"] >= int(session["fps"] / session["bufferSize"] * window_size_ms / 1000):
        # convert stream to numpy
        stream_bytes = [str.encode(i) if type(i) == str else i for i in session["frames"]]
        try:
            audio_data = np.frombuffer(b"".join(stream_bytes), dtype=np.int16).astype(np.float32) / audio_float_size
            # sound_data = np.frombuffer(b"".join(stream_bytes), dtype=np.int16).astype(np.float)
        except Exception as e:
            print(f"not able to read from buffer {e}")
            # reset
            session["windowSize"] = 0
            session["frames"] = []
            return
        # convert sample rate to 16K
        audio_data = librosa.resample(audio_data, session["fps"], sample_rate)
        print(audio_data.size)
        # for testing write to file
        # sf.write(f'{current_app.config["FILEDIR"]}temp.wav', audio_data, sample_rate)
        audio_data_length = audio_data.size / sample_rate * 1000
        # if given audio is less than window size, pad it
        if audio_data_length < window_size_ms:
            audio_data = np.append(audio_data, np.zeros(int(max_length - audio_data.size)))
        else:
            audio_data = audio_data[:max_length]
        # convert to tensor
        inp = torch.from_numpy(audio_data).float().to(device)
        # recording is stopped return
        if "batch" not in session:
            return
        session["batch"].append(inp)
        # reset
        session["windowSize"] = 0
        session["frames"] = []

    if len(session["batch"]) >= no_of_frames:
        audio_tensors = torch.stack(session["batch"])
        session["batch"] = []  # reset batch
        mel_audio_data = audio_transform(audio_tensors, device, sample_rate)
        zmuv_mel_audio_data = zmuv_transform(mel_audio_data)
        scores = model(zmuv_mel_audio_data.unsqueeze(1))
        scores = F.softmax(scores, -1).squeeze(1)

        idx = 0
        for score in scores:
            preds = score.cpu().detach().numpy()
            preds = preds / preds.sum()
            pred_idx = np.argmax(preds)
            pred_word = classes[pred_idx]
            print(f"predicted label {pred_idx} - {pred_word}")
            label = wake_words[session["target_state"]]
            if pred_word == label:
                session["target_state"] += 1
                session["infer_track"].append(pred_word)

                session["threads"] = []
                time_result = {}
                t = threading.Thread(target=plot_time, args=(time_result, audio_tensors[idx].numpy()))
                session["threads"].append(t)
                t.start()
                for th in session["threads"]:
                    th.join()

                session["threads"] = []
                mel_result = {}
                mels = audio_transform(audio_tensors[idx], device, sample_rate, skip_log=True)
                t = threading.Thread(target=plot_spectrogram, args=(mel_result, mels, "MelSpectrogram", "mel freq"))
                session["threads"].append(t)
                t.start()
                for th in session["threads"]:
                    th.join()

                session["threads"] = []
                log_mel_result = {}
                logmels = audio_transform(audio_tensors[idx], device, sample_rate)
                t = threading.Thread(
                    target=plot_spectrogram, args=(log_mel_result, logmels, "LogMelSpectrogram", "mel freq")
                )
                session["threads"].append(t)
                t.start()
                for th in session["threads"]:
                    th.join()

                word_details = {
                    "word": pred_word,
                    "buffer": audio_data.tolist(),
                    "time": time_result["plot"],
                    "mel": mel_result["plot"],
                    "logmel": log_mel_result["plot"],
                }
                emit("add-prediction", json.dumps(word_details))
                if session["infer_track"] == wake_words:
                    word_details = {"word": f"Wake word {' '.join(session['infer_track'])} detected"}
                    emit("add-prediction", json.dumps(word_details))
                    # reset
                    session["target_state"] = 0
                    session["infer_track"] = []
            idx = idx + 1


@socketio.on("end-recording", namespace="/audio")
def end_recording():
    """Stop recording audio from the client."""
    del session["frames"]
    del session["batch"]


if __name__ == "__main__":
    app.run(host="0.0.0.0")
    socketio.run(app)
