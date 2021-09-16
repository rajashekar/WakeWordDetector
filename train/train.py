import pandas as pd
from os import listdir
from os.path import isfile, join

import torch
import torch.nn as nn
import torch.utils.data as tud
from torch.optim.adamw import AdamW

from fastprogress import master_bar, progress_bar

from pathlib import Path

from tqdm import tqdm

from audio_collator import AudioCollator
from model import CNN
from transformers import ZmuvTransform, audio_transform

import numpy as np

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

wake_words = ["hey", "fourth", "brain"]
wake_words_sequence = ["0", "1", "2"]
wake_word_seq_map = dict(zip(wake_words, wake_words_sequence))

sr = 16000

wake_word_datapath = "wake_word_ds"

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print("CUDA is not available.  Training on CPU ...")
else:
    print("CUDA is available!  Training on GPU ...")

# load data
# Dataset checkpoint
positive_train_data = pd.read_csv(wake_word_datapath + "/positive/train.csv")
positive_dev_data = pd.read_csv(wake_word_datapath + "/positive/dev.csv")
positive_test_data = pd.read_csv(wake_word_datapath + "/positive/test.csv")

negative_train_data = pd.read_csv(wake_word_datapath + "/negative/train.csv")
negative_dev_data = pd.read_csv(wake_word_datapath + "/negative/dev.csv")
negative_test_data = pd.read_csv(wake_word_datapath + "/negative/test.csv")

train_ds = pd.concat([positive_train_data, negative_train_data]).sample(frac=1).reset_index(drop=True)
dev_ds = pd.concat([positive_dev_data, negative_dev_data]).sample(frac=1).reset_index(drop=True)
test_ds = pd.concat([positive_test_data, negative_test_data]).sample(frac=1).reset_index(drop=True)

# load generated data
train = pd.read_csv(wake_word_datapath + "/generated/train.csv")
dev = pd.read_csv(wake_word_datapath + "/generated/dev.csv")
test = pd.read_csv(wake_word_datapath + "/generated/test.csv")

train["timestamps"] = ""
train["duration"] = ""

dev["timestamps"] = ""
dev["duration"] = ""

test["timestamps"] = ""
test["duration"] = ""

train_ds = pd.concat([train_ds, train]).sample(frac=1).reset_index(drop=True)
dev_ds = pd.concat([dev_ds, dev]).sample(frac=1).reset_index(drop=True)
test_ds = pd.concat([test_ds, test]).sample(frac=1).reset_index(drop=True)

print(f"Training dataset size {train_ds.shape}")
print(f"Validation dataset size {dev_ds.shape}")
print(f"Test dataset size {test_ds.shape}")


def list_files(mypath):
    return [mypath + f for f in listdir(mypath) if isfile(join(mypath, f))]


noise_test = list_files("noise/noise_test/")
noise_train_complete = list_files("noise/noise_train/")

noise_train = noise_train_complete[: int(len(noise_train_complete) * 0.8)]
noise_dev = noise_train_complete[int(len(noise_train_complete) * 0.8) :]  # noqa

# random.randint(0,len(noise_dev))
# print noise data stats
print(f"Train noise dataset {len(noise_train)}")
print(f"Train noise dataset {len(noise_dev)}")
print(f"Train noise dataset {len(noise_test)}")


batch_size = 16
num_workers = 0

train_audio_collator = AudioCollator(noise_set=noise_train)
train_dl = tud.DataLoader(
    train_ds.to_dict(orient="records"),
    batch_size=batch_size,
    drop_last=True,
    shuffle=True,
    num_workers=num_workers,
    collate_fn=train_audio_collator,
)

dev_audio_collator = AudioCollator(noise_set=noise_dev)
dev_dl = tud.DataLoader(
    dev_ds.to_dict(orient="records"), batch_size=batch_size, num_workers=num_workers, collate_fn=dev_audio_collator
)

test_audio_collator = AudioCollator(noise_set=noise_test)
test_dl = tud.DataLoader(
    test_ds.to_dict(orient="records"), batch_size=batch_size, num_workers=num_workers, collate_fn=test_audio_collator
)

zmuv_audio_collator = AudioCollator()
zmuv_dl = tud.DataLoader(
    train_ds.to_dict(orient="records"), batch_size=1, num_workers=num_workers, collate_fn=zmuv_audio_collator
)

num_labels = len(wake_words) + 1  # oov
num_maps1 = 48
num_maps2 = 64
num_hidden_input = 768
hidden_size = 128
model = CNN(num_labels, num_maps1, num_maps2, num_hidden_input, hidden_size)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

zmuv_transform = ZmuvTransform().to(device)
if Path("zmuv.pt.bin").exists():
    zmuv_transform.load_state_dict(torch.load(str("zmuv.pt.bin")))
else:
    for idx, batch in enumerate(tqdm(zmuv_dl, desc="Constructing ZMUV")):
        zmuv_transform.update(batch["audio"].to(device))
    print(dict(zmuv_mean=zmuv_transform.mean, zmuv_std=zmuv_transform.std))
    torch.save(zmuv_transform.state_dict(), str("zmuv.pt.bin"))

print(f"Mean is {zmuv_transform.mean.item():0.6f}")
print(f"Standard Deviation is {zmuv_transform.std.item():0.6f}")

learning_rate = 0.001
weight_decay = 0.0001  # Weight regularization
lr_decay = 0.95

criterion = nn.CrossEntropyLoss()
params = list(filter(lambda x: x.requires_grad, model.parameters()))
optimizer = AdamW(params, learning_rate, weight_decay=weight_decay)

epochs = 20

# config for progress bar
mb = master_bar(range(epochs))
mb.names = ["Training loss", "Validation loss"]
x = []

training_losses = []
validation_losses = []

valid_mean_min = np.Inf

for epoch in mb:
    x.append(epoch)
    # Evaluate
    model.train()
    total_loss = torch.Tensor([0.0]).to(device)
    # pbar = tqdm(train_dl, total=len(train_dl), position=0, desc="Training", leave=True)
    for batch in progress_bar(train_dl, parent=mb):
        audio_data = batch["audio"].to(device)
        labels = batch["labels"].to(device)
        # get mel spectograms
        mel_audio_data = audio_transform(audio_data)
        # do zmuv transform
        mel_audio_data = zmuv_transform(mel_audio_data)
        predicted_scores = model(mel_audio_data.unsqueeze(1))
        # get loss
        loss = criterion(predicted_scores, labels)

        optimizer.zero_grad()
        model.zero_grad()

        # backward propagation
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            total_loss += loss

    for group in optimizer.param_groups:
        group["lr"] *= lr_decay

    mean = total_loss / len(train_dl)
    training_losses.append(mean.cpu())

    # Evaluate
    model.eval()
    validation_loss = torch.Tensor([0.0]).to(device)
    with torch.no_grad():
        # pbar = tqdm(dev_dl, total=len(dev_dl), position=0, desc="Evaluating", leave=True)
        for batch in progress_bar(dev_dl, parent=mb):
            audio_data = batch["audio"].to(device)
            labels = batch["labels"].to(device)
            # get mel spectograms
            mel_audio_data = audio_transform(audio_data)
            # do zmuv transform
            mel_audio_data = zmuv_transform(mel_audio_data)
            predicted_scores = model(mel_audio_data.unsqueeze(1))
            # get loss
            loss = criterion(predicted_scores, labels)
            validation_loss += loss

    val_mean = validation_loss / len(dev_dl)
    validation_losses.append(val_mean.cpu())

    # Update training chart
    mb.update_graph([[x, training_losses], [x, validation_losses]], [0, epochs])
    mb.write(
        f"\nEpoch {epoch}: Training loss {mean.item():.6f}"
        + " validation loss {val_mean.item():.6f} with lr {group['lr']:.6f}"
    )

    # save model if validation loss has decreased
    if val_mean.item() <= valid_mean_min:
        print(
            "Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...".format(valid_mean_min, val_mean.item())
        )
        torch.save(model.state_dict(), "model_hey_fourth_brain_best.pt")
        valid_mean_min = val_mean.item()


# track test loss
test_loss = 0.0
classes = wake_words[:]
# oov
classes.append("oov")
class_correct = list(0.0 for i in range(len(classes)))
class_total = list(0.0 for i in range(len(classes)))

actual = []
predictions = []

model.eval()
# iterate over test data
pbar = tqdm(test_dl, total=len(test_dl), position=0, desc="Testing", leave=True)
for batch in pbar:
    # move tensors to GPU if CUDA is available
    audio_data = batch["audio"].to(device)
    labels = batch["labels"].to(device)
    # forward pass: compute predicted outputs by passing inputs to the model
    mel_audio_data = audio_transform(audio_data)
    # do zmuv transform
    mel_audio_data = zmuv_transform(mel_audio_data)
    output = model(mel_audio_data.unsqueeze(1))
    # calculate the batch loss
    loss = criterion(output, labels)
    # update test loss
    test_loss += loss.item() * audio_data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)
    # compare predictions to true label
    correct_tensor = pred.eq(labels.data.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    # calculate test accuracy for each object class
    for i in range(labels.shape[0]):
        label = labels.data[i]
        class_correct[label.long()] += correct[i].item()
        class_total[label.long()] += 1
        # for confusion matrix
        actual.append(classes[labels.data[i].long().item()])
        predictions.append(classes[pred.data[i].item()])

# plot confusion matrix
cm = confusion_matrix(actual, predictions, labels=classes)
print(classification_report(actual, predictions))
cmp = ConfusionMatrixDisplay(cm, classes)
# fig, ax = plt.subplots(figsize=(8, 8))
# cmp.plot(ax=ax, xticks_rotation="vertical")


# average test loss
test_loss = test_loss / len(test_ds)
print("Test Loss: {:.6f}\n".format(test_loss))

for i in range(len(classes)):
    if class_total[i] > 0:
        print(
            "Test Accuracy of %5s: %2d%% (%2d/%2d)"
            % (classes[i], 100 * class_correct[i] / class_total[i], np.sum(class_correct[i]), np.sum(class_total[i]))
        )
    else:
        print("Test Accuracy of %5s: N/A (no training examples)" % (classes[i]))

print(
    "\nTest Accuracy (Overall): %2d%% (%2d/%2d)"
    % (100.0 * np.sum(class_correct) / np.sum(class_total), np.sum(class_correct), np.sum(class_total))
)
