import os
import sys

i_process, n_process, n_sample = int(sys.argv[-3]), int(sys.argv[-2]), int(sys.argv[-1])
i_sample = n_sample // n_process
sys.path.append(os.getcwd())

name_model = "model_colon_exp_1"
c = None
exec('import models.' + name_model + "_config as c")

from src.inn_model import *
from src.dataset import *
from src.visualisation import *
from tqdm import tqdm

import glob
import copy
import torch
import pandas as pd
import numpy as np

from hvs_hsi_pytorch.utils import hsi_utils, utils
from hvs_hsi_pytorch.hs_image import HSImage

# Read reformatted dataset
data_colon = pd.read_csv("data/mc/colon_train_15.7.15.4_power_led_300.csv")
train_df = data_colon[:int(len(data_colon) * 0.9)]

# Data loader
train_dataset = McDataset(train_df, c)
train_loader = data_utils.DataLoader(train_dataset, batch_size=c.batch_size, shuffle=True, drop_last=True)

# Set up dimensionality
c.ndim_y = train_dataset.get_y_dim()
c.ndim_x = train_dataset.get_x_dim()
c.ndim_tot = max(c.ndim_y, c.ndim_x) + c.ndim_z

# Model
model = inn_model(c)
model.load_state_dict(torch.load('models/' + name_model, map_location=torch.device(c.device)))
_ = model.eval()

# Read HSI image data

# img_path = 'data/finger_data/session_001/P010001_V03_iHSI_T01_2022-07-28-16-16-54/raw/frame_000181_raw.png'
# white_path = 'data/finger_data/session_001/exported_white_2022-07-15-20-29-45.png'
# img = hsi_utils.load_frame(img_path)
# white = hsi_utils.load_sample(white_path)
# hsi_data = HSImage(array=img, wavelengths=None, camera="imec 4x4-VIS-15.7.15.4")
# hsi_data.reconstruct(white=white, dark=None, rho=15, method="flatfield")
# hsi_data.array = utils.normalise_L1(hsi_data.array)

img_path = "data/imgs/frames/exported1465.png"
white_path = "data/imgs/exported_white_2022-03-10-10-45-04.png"
img = hsi_utils.load_sample(img_path)
white = hsi_utils.load_sample(white_path)
hsi_data = HSImage(array=img, wavelengths=None, camera="imec 4x4-VIS-15.7.15.4")
hsi_data.calibrate(white=white)
hsi_data.demosaic()
hsi_data.correct()
hsi_data.array = utils.normalise_L1(hsi_data.array)

data = hsi_data.array
shape = data.shape
data = data.reshape(shape[0], shape[1] * shape[2]).swapaxes(0, 1)

step = 2 ** 13
batch = list(np.arange(data.shape[0], step=step))
batch.append(data.shape[0])

all_output = torch.tensor([])
with open('runs/run' + sys.argv[-3] + '.txt', 'w') as fifo:
    count = 0
    fifo.write(str(count) + '\n')
    fifo.flush()

    for j in range(i_sample):
        output = torch.tensor([])
        for i in range(len(batch) - 1):
            y = data[batch[i]:batch[i + 1]].to(c.device)
            z = torch.randn(y.shape[0], c.ndim_z).to(c.device)
            y = torch.cat((torch.zeros(y.shape[0], c.ndim_tot - c.ndim_y - c.ndim_z).to(c.device), y), dim=1)
            y = torch.cat((z, y), dim=1).to(c.device)
            pred_x = model(y, rev=True)[0][:, c.target].detach().cpu().clone()
            output = torch.cat((output, pred_x))

        all_output = torch.cat([all_output, output[:, None]], dim=1)
        count += 1
        fifo.write(str(count) + '\n')
        fifo.flush()

torch.save(all_output, 'tensors/tensor' + sys.argv[-3] + '.pt')
exit(0)
