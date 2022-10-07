# Import
import os
import torch
import pandas as pd
import numpy as np

from hvs_hsi_pytorch.hsicamera_simulator import HSICameraSimulator
from hvs_hsi_pytorch.hs_image import HSImage
from hvs_hsi_pytorch.utils.utils import normalise_L1
from hvs_hsi_pytorch.definitions import DIR_DATA


# Read reformatted dataset
file_name = 'generic_train'
mc_path = os.path.join(DIR_DATA, "monte_carlo/dkfz/", file_name + ".csv")
df_reflectance = pd.read_csv(mc_path, skiprows=[0], usecols=list(range(26, 171)))
wavelengths_origin = torch.tensor([eval(i.split('_')[-1]) * 1e9 for i in df_reflectance.columns.to_list()])
df_reflectance.columns = wavelengths_origin.numpy()

df_tissu = pd.read_csv(mc_path, usecols=list(range(2, 26)), low_memory=False)
df_tissu.columns = [str(i) + '_' + str(j) for i, j in zip(df_tissu.columns, df_tissu.iloc[0, :].to_list())]
df_tissu = df_tissu.drop(0)

# Reshape data into a virtual HSI image
reflectance_array = torch.tensor(df_reflectance.to_numpy()[:, None], dtype=torch.float32)
reflectance_array = torch.swapaxes(reflectance_array, 0, 2)
reflectance_array = torch.swapaxes(reflectance_array, 1, 2)
hsi_image = HSImage(reflectance_array, wavelengths=wavelengths_origin, camera=None)

# Transformation
imec_camera = HSICameraSimulator(camera="imec 4x4-VIS-15.7.15.4")
hsi_image_imec = imec_camera(hsi_image)
hsi_image_imec.array = normalise_L1(hsi_image_imec.array)
wavelengths_output = hsi_image_imec.wavelengths

reflectance_output = torch.squeeze(torch.swapaxes(hsi_image_imec.array, 0, 2))
target_param = torch.tensor(df_tissu.to_numpy(dtype=np.float32))
reformatted_data = torch.cat((target_param, reflectance_output), dim=1).numpy()
columns = df_tissu.columns.to_list() + list(map(str, wavelengths_output.numpy()))
df_output = pd.DataFrame(reformatted_data, columns=columns)
new_file_name = file_name + "_15.7.15.4_norm_l1.csv"
df_output.to_csv(os.path.join('./data/mc', new_file_name), index=False)
