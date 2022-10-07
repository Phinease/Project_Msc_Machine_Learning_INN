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

type_model = "generic"
nb_model = "0613"
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

c.parameters = ["sao2", "vhb"]
c.n_spectrum_start = 24
c.target = 0

# Data loader
train_dataset = McDataset(train_df, c)
train_loader = data_utils.DataLoader(train_dataset, batch_size=c.batch_size, shuffle=True, drop_last=True)

# Set up dimensionality
c.ndim_y = train_dataset.get_y_dim()
c.ndim_x = train_dataset.get_x_dim()
c.ndim_tot = max(c.ndim_y, c.ndim_x) + c.ndim_z

# Model
model = inn_model_1(c)
model.load_state_dict(torch.load("models/" + name_model, map_location=torch.device(c.device)))
_ = model.eval()

white_path = "data/imgs/exported_white_2022-03-10-10-45-04.png"
img_files = glob.glob("data/imgs/frames/*")

white = hsi_utils.load_sample(white_path)
for img_path in tqdm(img_files, ascii=True, ncols=100):
    save_name = img_path.split('/')[-1].split('.')[-2] + "_" + type_model + "_" + nb_model
    img = hsi_utils.load_sample(img_path)
    hsi_data = HSImage(array=img, wavelengths=None, camera="imec 4x4-VIS-15.7.15.4")
    hsi_data.calibrate(white=white)
    hsi_data.demosaic()
    hsi_data.correct()
    hsi_origin = copy.deepcopy(hsi_data)

    data = normalise_L1(hsi_data.array)
    shape = data.shape
    data = data.reshape(shape[0], shape[1] * shape[2]).swapaxes(0, 1)

    step = 2 ** 15
    batch = list(np.arange(data.shape[0], step=step))
    batch.append(data.shape[0])

    output = torch.tensor([])
    for i in range(len(batch) - 1):
        y = data[batch[i]:batch[i + 1]].to(c.device)
        z = torch.randn(y.shape[0], c.ndim_z).to(c.device)
        y = torch.cat((torch.zeros(y.shape[0], c.ndim_tot - c.ndim_y - c.ndim_z).to(c.device), y), dim=1)
        y = torch.cat((z, y), dim=1).to(c.device)
        pred_x = model(y, rev=True)[0][:, c.target].detach().cpu().clone()
        output = torch.cat((output, pred_x))

    output = output.reshape(shape[1], shape[2])[:, :, None]

    # show_hvs_combo(hsi_origin, save_name=save_name)
    # show_inn_combo_origin(hsi_origin, output, save_name=save_name)
    # show_inn_combo_norm(hsi_origin, output, save_name=save_name)
    show_inn_combo_range(hsi_origin, output, aim_range=(-0.2, 1.2), save_name=save_name)
    plt.close('all')