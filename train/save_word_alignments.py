import os

import pandas as pd
import textgrid

wake_word_datapath = "wake_word_ds"


def get_timestamps(path):
    filename = path.split("/")[-1].split(".")[0]
    filepath = f"aligned_data/audio/{filename}.TextGrid"
    words_timestamps = {}
    if os.path.exists(filepath):
        tg = textgrid.TextGrid.fromFile(filepath)
        for tg_intvl in range(len(tg[0])):
            word = tg[0][tg_intvl].mark
            if word:
                words_timestamps[word] = {"start": tg[0][tg_intvl].minTime, "end": tg[0][tg_intvl].maxTime}
    return words_timestamps


positive_train_data = pd.read_csv("positive/train.csv")
positive_dev_data = pd.read_csv("positive/dev.csv")
positive_test_data = pd.read_csv("positive/test.csv")

negative_train_data = pd.read_csv("negative/train.csv")
negative_dev_data = pd.read_csv("negative/dev.csv")
negative_test_data = pd.read_csv("negative/test.csv")

positive_train_data["path"] = positive_train_data["path"].apply(lambda x: "positive/audio/" + x.split(".")[0] + ".wav")
positive_dev_data["path"] = positive_dev_data["path"].apply(lambda x: "positive/audio/" + x.split(".")[0] + ".wav")
positive_test_data["path"] = positive_test_data["path"].apply(lambda x: "positive/audio/" + x.split(".")[0] + ".wav")

negative_train_data["path"] = negative_train_data["path"].apply(lambda x: "negative/audio/" + x.split(".")[0] + ".wav")
negative_dev_data["path"] = negative_dev_data["path"].apply(lambda x: "negative/audio/" + x.split(".")[0] + ".wav")
negative_test_data["path"] = negative_test_data["path"].apply(lambda x: "negative/audio/" + x.split(".")[0] + ".wav")

positive_train_data["timestamps"] = positive_train_data["path"].progress_apply(get_timestamps)
positive_dev_data["timestamps"] = positive_dev_data["path"].progress_apply(get_timestamps)
positive_test_data["timestamps"] = positive_test_data["path"].progress_apply(get_timestamps)

negative_train_data["timestamps"] = negative_train_data["path"].progress_apply(get_timestamps)
negative_dev_data["timestamps"] = negative_dev_data["path"].progress_apply(get_timestamps)
negative_test_data["timestamps"] = negative_test_data["path"].progress_apply(get_timestamps)

# save above data
positive_train_data.to_csv(wake_word_datapath + "/positive/train.csv", index=False)
positive_dev_data.to_csv(wake_word_datapath + "/positive/dev.csv", index=False)
positive_test_data.to_csv(wake_word_datapath + "/positive/test.csv", index=False)

negative_train_data.to_csv(wake_word_datapath + "/negative/train.csv", index=False)
negative_dev_data.to_csv(wake_word_datapath + "/negative/dev.csv", index=False)
negative_test_data.to_csv(wake_word_datapath + "/negative/test.csv", index=False)
