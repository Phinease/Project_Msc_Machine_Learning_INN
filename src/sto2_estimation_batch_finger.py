import os
import sys
import glob

sys.path.append(os.getcwd())

from inn_model import *
from dataset import *
from visualisation import *

from tqdm import tqdm
from hvs_hsi_pytorch.utils import hsi_utils
from hvs_hsi_pytorch.hs_image import HSImage
from hvs_hsi_pytorch.utils.utils import normalise_L1


type_model = "pig"
nb_model = "2"
name_model = "model_" + type_model + "_" + nb_model
c = None
exec("import models." + name_model + "_config as c")


def clear():
    os.system('clear')


# Read reformatted dataset
data_colon = pd.read_csv("data/mc/generic_train_15.7.15.4_power_led_300_norm_l1.csv")

N = len(data_colon)
n_train = int(N * 0.9)
train_df = data_colon[:n_train]

c.target = 0

# Data loader
train_dataset = McDataset(train_df, c)
train_loader = data_utils.DataLoader(train_dataset, batch_size=c.batch_size, shuffle=True, drop_last=True)

# Set up dimensionality
c.ndim_y = train_dataset.get_y_dim()
c.ndim_x = train_dataset.get_x_dim()
c.ndim_tot = max(c.ndim_y, c.ndim_x) + c.ndim_z
n_sample = 10

# Model
model = inn_model(c)
model.load_state_dict(torch.load("models/" + name_model, map_location=torch.device(c.device)))
_ = model.eval()

white_path = "data/finger_data/session_001/exported_white_2022-07-15-20-29-45.png"
img_files = glob.glob("data/finger_data/session_001/P010001_V03_iHSI_T01_2022-07-28-16-16-54/raw/*")

white = hsi_utils.load_sample(white_path)
for img_path in tqdm(img_files, ascii=True, ncols=100):
    save_name = img_path.split('/')[-1].split('_')[1] + "_" + type_model + "_" + nb_model + "_" + str(n_sample)
    img = hsi_utils.load_frame(img_path, bitshift=False)
    hsi_origin = HSImage(array=copy.deepcopy(img), wavelengths=None, camera="imec 4x4-VIS-15.7.15.4")
    # hsi_origin.reconstruct(white=copy.deepcopy(white), dark=None, rho=15, method="flatfield")
    hsi_origin.reconstruct(white=None, dark=None,  rho=0.88, lightsource='d_light_c', method='spectral')

    hsi_data = HSImage(array=img, wavelengths=None, camera="imec 4x4-VIS-15.7.15.4")
    # hsi_data.reconstruct(white=white, dark=None, rho=0.88, method="flatfield")
    hsi_data.reconstruct(white=None, dark=None,  rho=0.88, lightsource='d_light_c', method='spectral')

    data = normalise_L1(hsi_data.array)
    shape = data.shape
    data = data.reshape(shape[0], shape[1] * shape[2]).swapaxes(0, 1)
    output_all = torch.zeros(size=(n_sample, shape[1] * shape[2]))

    step = 2 ** 18
    batch = list(np.arange(data.shape[0], step=step))
    batch.append(data.shape[0])

    # for j in range(n_sample):
    #     output = torch.tensor([])
    #     for i in range(len(batch) - 1):
    #         y = data[batch[i]:batch[i + 1]].to(c.device)
    #         z = torch.randn(y.shape[0], c.ndim_z).to(c.device)
    #         y = torch.cat((torch.zeros(y.shape[0], c.ndim_tot - c.ndim_y - c.ndim_z).to(c.device), y), dim=1)
    #         y = torch.cat((z, y), dim=1).to(c.device)
    #         pred_x = model(y, rev=True)[0][:, c.target].detach().cpu().clone()
    #         output = torch.cat((output, pred_x))
    #     output_all[j] = output

    output = torch.tensor([])
    for i in range(len(batch) - 1):
        y = data[batch[i]:batch[i + 1]].to(c.device)
        z = torch.randn(y.shape[0], c.ndim_z).to(c.device)
        y = torch.cat((torch.zeros(y.shape[0], c.ndim_tot - c.ndim_y - c.ndim_z).to(c.device), y), dim=1)
        y = torch.cat((z, y), dim=1).to(c.device)
        pred_x = model(y, rev=True)[0][:, c.target].detach().cpu().clone()
        output = torch.cat((output, pred_x))
    output = output.reshape(shape[1], shape[2])[:, :, None]

    # show_hvs_combo(hsi_origin, mask=False, save_name=save_name)
    show_inn_combo_norm(hsi_origin, output, mask=False, save_name=save_name)
    # show_inn_combo_origin(hsi_origin, output, mask=False, aim_range=(-0.1, 1.1), save_name=save_name)
    plt.close('all')

