import re

import pandas as pd
import os

folder_path = 'aug_data'
train = pd.DataFrame()
val = pd.DataFrame()

for sub_dir in os.listdir(folder_path):
    full_path = os.path.join(folder_path, sub_dir)
    videos = [full_path + "/" + i for i in os.listdir(full_path)]
    # label = int(sub_dir.split("_")[1]) -1
    label = int(re.split('(\d+)', sub_dir)[1]) - 1
    num_data = int(len(videos) * 0.8)

    train_data = {'path': videos[:num_data], 'label': label}
    val_data = {'path': videos[num_data:], 'label': label}

    train = train.append(pd.DataFrame(train_data), ignore_index=True)
    val = val.append(pd.DataFrame(val_data), ignore_index=True)

train = train.sample(frac=1).reset_index(drop=True)
val = val.sample(frac=1).reset_index(drop=True)

train.to_csv("frame_list/train.csv", index=False, sep=' ', header=None)
val.to_csv("frame_list/val.csv", index=False, sep=' ', header=None)


# python tools/run_net.py
# --cfg configs/Kinetics/C2D_8x8_R50.yaml TRAIN.BATCH_SIZE 8
# DATA.PATH_TO_DATA_DIR /home/softmaxai/Bhavika_Kanani/TechMahindra/data/frame_list
