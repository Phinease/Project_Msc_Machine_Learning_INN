import os
import torch
import pandas as pd
import numpy as np

from hvs_hsi_pytorch.hs_image import HSImage
from hvs_hsi_pytorch.hsicamera_simulator import HSICameraSimulator
from hvs_hsi_pytorch.utils.utils import normalise_L1
from hvs_hsi_pytorch.definitions import DIR_DATA

CAMERAS = [
    '15.7.10.12',
    '15.7.14.2',
    '15.7.15.4',
    '15.7.16.7'
]

LIGHTS = [
    'asahi',
    'd_light_c',
    'power_led_300',
    'sunoptic_x450',
    'torch',
    'xenon_nova_300'
]

i_camera = 0
i_light = 2

# Read dataset
file_name = 'generic_train'
mc_path = os.path.join(DIR_DATA, "monte_carlo/dkfz/", file_name + ".csv")
df_reflectance = pd.read_csv(mc_path, skiprows=[0], usecols=list(range(26, 171)))

wavelengths_origin = torch.tensor([eval(i.split('_')[-1]) * 1e9 for i in df_reflectance.columns.to_list()])
df_reflectance.columns = wavelengths_origin.numpy()

df_tissu = pd.read_csv(mc_path, usecols=list(range(2, 26)), low_memory=False)
df_tissu.columns = [str(i) + '_' + str(j) for i, j in zip(df_tissu.columns, df_tissu.iloc[0, :].to_list())]
df_tissu = df_tissu.drop(0)

# Reshape data into a HSI image
reflectance_array = torch.tensor(df_reflectance.to_numpy()[:, None], dtype=torch.float32)
reflectance_array = torch.swapaxes(reflectance_array, 0, 2)
reflectance_array = torch.swapaxes(reflectance_array, 1, 2)
hsi_image = HSImage(reflectance_array, wavelengths=wavelengths_origin, camera=None)

imec_camera_lightsource = HSICameraSimulator(camera="imec 4x4-VIS-" + CAMERAS[i_camera],
                                             simulate_snapshot=False,
                                             simulate_lightsource=True,
                                             lightsource=LIGHTS[i_light])
imec_image_lightsource = imec_camera_lightsource(hsi_image)
# imec_image_lightsource.array = normalise_L1(imec_image_lightsource.array)

# Save output dataframe
reflectance_camera_response = torch.squeeze(torch.swapaxes(imec_image_lightsource.array, 0, 2))
target_param = torch.tensor(df_tissu.to_numpy(dtype=np.float32))
reformatted_data = torch.cat((target_param, reflectance_camera_response), dim=1).numpy()
columns = df_tissu.columns.to_list() + list(map(str, imec_image_lightsource.wavelengths.numpy()))
df_output = pd.DataFrame(reformatted_data, columns=columns)
new_file_name = file_name + "_" + CAMERAS[i_camera] + "_" + LIGHTS[i_light] + ".csv"
# new_file_name = file_name + "_" + CAMERAS[i_camera] + "_" + LIGHTS[i_light] + "_norm_l1.csv"

df_output.to_csv(os.path.join('./data/mc', new_file_name), index=False)
