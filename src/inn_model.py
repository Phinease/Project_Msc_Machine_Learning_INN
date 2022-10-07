from time import time

import os
import glob
import logging
import torch.nn as nn
from tqdm import tqdm

from sklearn import metrics
from IPython.display import clear_output

from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet
from FrEIA.modules import *

from src.visualisation import *

from hvs_hsi_pytorch.utils import hsi_utils, utils
from hvs_hsi_pytorch.hs_image import HSImage


def inn_model(c):
    """
    Construct INN model

    :param c: config.py
    :return: ReversibleGraphNet model
    """

    def subnet_fc(c_in, c_out):
        return nn.Sequential(nn.Linear(c_in, c.n_neurons), nn.ReLU(),
                             nn.Linear(c.n_neurons, c_out))

    nodes = [InputNode(c.ndim_tot, name='input')]

    for k in range(c.n_layers):
        nodes.append(Node(nodes[-1],
                          c.block,
                          {'subnet_constructor': subnet_fc, 'clamp': 2.0},
                          name=F'coupling_{k}'))
        if c.block != GLOWCouplingBlock:
            nodes.append(Node(nodes[-1],
                            PermuteRandom,
                            {'seed': k},
                            name=F'permute_{k}'))

    nodes.append(OutputNode(nodes[-1], name='output'))

    return ReversibleGraphNet(nodes, verbose=False).to(c.device)

    
def train(model, train_loader, valid_loader, c):
    if os.path.exists("runs/current.log"):
        os.remove("runs/current.log")
    file_handler = logging.FileHandler(filename='runs/current.log', mode='a')

    logger = logging.getLogger()
    c.logger = logger
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    t_start = time()
    train_mae, valid_mae = [], []
    try:
        # print("Training")
        for i_epoch in range(c.n_epochs):
            i, j = train_per_epoch(model, train_loader, valid_loader, c, logger, i_epoch=i_epoch)
            
            train_mae.append(i)
            valid_mae.append(j)

            if j < c.early_stop_mse:
                break
    except KeyboardInterrupt:
        pass
    finally:
        plt.figure()
        fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(12, 4))
        
        print(f"\n\nTraining took {(time() - t_start) / 60:.2f} minutes\n")
        with torch.no_grad():
            axs.set_ylim([0., 0.5])
            axs.plot(range(len(train_mae) - 2), train_mae[2:], c='blue', label='Train MAE')
            axs.plot(range(len(valid_mae) - 2), valid_mae[2:], c='orange', label='Valid MAE')
            axs.set_xlabel('Epoch')
            axs.set_ylabel('MAE')
            axs.legend()

        fig.savefig('runs/current_cross_valid.png', dpi=300)
        plt.show()


def train_per_epoch(model, data_loader, valid_loader, c, logger, i_epoch=0):
    """
    Train forward and backward for one epoch

    :param valid_loader: torch.utils.data.DataLoader
    :param c: config.py
    :param logger: logging.loger
    :param model: InnModel <- module.nn
    :param data_loader: torch.utils.data.DataLoader
    :param i_epoch: int
    :return:
    """

    t_start = time()
    model.train()

    # If MMD on x-space is present from the start, the model can get stuck.
    # Instead, ramp it up exponentially.
    loss_factor = min(1., 2. * 0.002 ** (1. - (float(i_epoch) / c.n_epochs)))
    loss_tot, normalizer, batch_idx = 0, 0, 0
    train_mae = []
    loss_record = torch.zeros(size=(4, ))

    for x, y in data_loader:
        batch_losses = []
        normalizer += len(x)
        batch_idx += 1
        if c.n_its_per_epoch is not None and batch_idx > c.n_its_per_epoch:
            break

        c.optimizer.zero_grad()

        x, y = x.to(c.device), y.to(c.device)
        y_clean = y.clone()
        x_clean = x.clone()

        # Forward step:
        # (x, pad_x) -> (z, pad_z_y, y)

        pad_x = c.zeros_noise_scale * torch.randn(c.batch_size, c.ndim_tot - c.ndim_x, device=c.device)
        pad_z_y = c.zeros_noise_scale * torch.randn(c.batch_size, c.ndim_tot - c.ndim_y - c.ndim_z, device=c.device)

        x = torch.cat((x, pad_x), dim=1)
        z = torch.randn(c.batch_size, c.ndim_z, device=c.device)
        y += c.y_noise_scale * torch.randn(c.batch_size, c.ndim_y, dtype=torch.float, device=c.device)
        y = torch.cat((z, pad_z_y, y), dim=1)
        y_short = torch.cat((y[:, :c.ndim_z], y[:, -c.ndim_y:]), dim=1)

        output = model(x)

        batch_losses.append(c.lambda_fit * c.loss_fit(output[0][:, c.ndim_z:], y[:, c.ndim_z:], c))
        output_block_grad = torch.cat((output[0][:, :c.ndim_z], output[0][:, -c.ndim_y:].data), dim=1)
        batch_losses.append(c.lambda_latent * c.loss_latent(output_block_grad, y_short, c))

        # temp = sum(batch_losses)
        # temp.backward()
        # batch_losses = []

        # Backward step:
        # (z, pad_z_y, y) -> (x, pad_x)

        pad_z_y = c.zeros_noise_scale * torch.randn(c.batch_size, c.ndim_tot - c.ndim_y - c.ndim_z, device=c.device)
        y = y_clean + c.y_noise_scale * torch.randn(c.batch_size, c.ndim_y, device=c.device)
        orig_z_perturbed = (output[0].data[:, :c.ndim_z] + c.y_noise_scale * torch.randn(c.batch_size, c.ndim_z, device=c.device))
        y_rev = torch.cat((orig_z_perturbed, pad_z_y, y), dim=1)
        y_rev_rand = torch.cat((torch.randn(c.batch_size, c.ndim_z, device=c.device), pad_z_y, y), dim=1)

        output_rev = model(y_rev, rev=True)
        output_rev_rand = model(y_rev_rand, rev=True)

        batch_losses.append(c.lambda_reconstruct * c.loss_reconstruct(output_rev[0][:, :c.ndim_x], x[:, :c.ndim_x], c))
        batch_losses.append(c.lambda_backward * c.loss_backward(output_rev_rand[0][:, :c.ndim_x], x[:, :c.ndim_x], c))

        temp = sum(batch_losses)
        temp.backward()
        loss_record += torch.tensor(batch_losses)

        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.clamp_(-15.00, 15.00)

        c.optimizer.step()
        train_mae.append(c.metric_mae(output_rev_rand[0][:, c.target], x_clean[:, c.target]))

    # if total_loss.isnan() or total_loss_rev.isnan() or total_loss_tot.isnan():
    #     raise ExplodeLoss

    train_mae, valid_mae = torch.tensor(train_mae).mean(), None
    for valid_x, valid_y in valid_loader:
        pred_x = predict(model, valid_y, c)
        valid_mae = c.metric_mae(valid_x[:, c.target], pred_x[:, c.target])
        break

    if c.verbose == 1:
        info = "Epoch %d: time - %.2f | LOSS forward mse mmd backward mmd mse " + str(loss_record)
        logger.info(info)
        clear_output(wait=True)
        print("Epoch", i_epoch, ":")
        plot_valid(model, valid_loader, c)

    return train_mae, valid_mae


def predict(model, y, c):
    """
    Compute prediction of a group of y on Inn model (cuda optimized)Layer

    :param c: config.py
    :param model: Inn model
    :param y: torch.Tensor
    :return: torch.Tensor
    """

    model.eval()
    y = y.to(c.device)
    test_size = y.shape[0]
    pad_z_y = c.zeros_noise_scale * torch.randn(test_size, c.ndim_tot - c.ndim_y - c.ndim_z, device=c.device)
    y = y + c.y_noise_scale * torch.randn(test_size, c.ndim_y, device=c.device)
    z = torch.randn(test_size, c.ndim_z, device=c.device)
    y = torch.cat((z, pad_z_y, y), dim=1)

    return model(y, rev=True)[0].detach().cpu()


def compute_posterior(model, y, z, c):
    """
    Compute posterior of x of specific y on INN model (cuda optimized)

    :param c: config.py
    :param z: torch.Tensor
    :param model: Inn model
    :param y: torch.Tensor
    :return: estimation of x
    """

    model.eval()
    z = z.to(c.device)
    y = y.unsqueeze(0).expand(z.shape[0], -1).to(c.device)
    y = torch.cat((torch.zeros(y.shape[0], c.ndim_tot - c.ndim_y - c.ndim_z).to(c.device), y), dim=1)
    y = torch.cat((z, y), dim=1)

    return model(y, rev=True)[0].detach().cpu()


def compute_posterior_2(model, y, z, c):
    """
    Compute posterior of x of specific y on INN model (cuda optimized)

    :param c: config.py
    :param z: torch.Tensor
    :param model: Inn model
    :param y: torch.Tensor
    :return: estimation of x
    """

    z = z.to(c.device)
    y = y.to(c.device)
    pad_z_y = torch.zeros((y.shape[0], c.ndim_tot - c.ndim_y - c.ndim_z), device=c.device)
    y = torch.cat((z, pad_z_y, y), dim=1)

    return model(y, rev=True)[0].detach().cpu()


class ExplodeLoss(Exception):
    """
    Raise when INN model's loss if NaN or Inf
    """

    pass


def plot_valid(model, valid_loader, c):
    plt.figure()
    fig, axs = plt.subplots(3, 3, constrained_layout=True, figsize=(12, 9))

    for x, y in valid_loader:
        model.eval()
        pred_x = predict(model, y, c)
        ax = axs.flatten()
        with torch.no_grad():
            for i, j in enumerate(c.targets[:min(9, len(c.targets))]):
                ax_min, ax_max = x[:, i].min(), x[:, i].max()
                ax[i].set_title('Layer ' + str(j))
                ax[i].scatter(x[:, i], pred_x[:, i], alpha=0.2)
                ax[i].plot(np.linspace(ax_min, ax_max, 100), np.linspace(ax_min, ax_max, 100), c='red')
                extend = ax_max - ax_min
                ax[i].set_ylim([ax_min - extend * 0.25, ax_max + extend * 0.25])
                ax[i].set_xlim([ax_min - extend * 0.1, ax_max + extend * 0.1])
            info = "MAE Sao2 %d - %f" % (
                c.target, metrics.mean_absolute_error(x[:, c.target].numpy(), pred_x[:, c.target].numpy()))
            print(info)
            c.logger.info(info)
        break
    fig.savefig('runs/current_valid.png', dpi=300)
    plt.show()


def plot_colon_estimation(model, c):
    # Colon Test estimation
    # Read HSI image data
    img_path = "data/imgs/frames/exported1465.png"
    white_path = "data/imgs/exported_white_2022-03-10-10-45-04.png"

    img = hsi_utils.load_sample(img_path)
    white = hsi_utils.load_sample(white_path)

    hsi_data = HSImage(array=img, wavelengths=None, camera="imec 4x4-VIS-15.7.15.4")
    hsi_data.calibrate(white=white)
    hsi_data.demosaic()
    hsi_data.correct()
    hsi_origin = copy.deepcopy(hsi_data)
    hsi_data.array = utils.normalise_L1(hsi_data.array)

    data = hsi_data.array
    shape = data.shape
    data = data.reshape(shape[0], shape[1] * shape[2]).swapaxes(0, 1)

    step = 2 ** 13
    batch = list(np.arange(data.shape[0], step=step))
    batch.append(data.shape[0])

    output = torch.tensor([])
    for i in tqdm(range(len(batch) - 1), ascii=True, ncols=100):
        y = data[batch[i]:batch[i + 1]].to(c.device)
        z = torch.randn(y.shape[0], c.ndim_z).to(c.device)
        y = torch.cat((torch.zeros(y.shape[0], c.ndim_tot - c.ndim_y - c.ndim_z).to(c.device), y), dim=1)
        y = torch.cat((z, y), dim=1).to(c.device)
        pred_x = model(y, rev=True)[0][:, c.target].detach().cpu().clone()
        output = torch.cat((output, pred_x))

    output = output.reshape(shape[1], shape[2])[:, :, None]

    show_inn_combo_norm(hsi_origin, output, mask=True)


def plot_finger_estimation(model, c):
    # Read HSI image data
    img_path = "data/finger_data/session_001/P010001_V03_iHSI_T01_2022-07-28-16-16-54/raw/frame_000121_raw.png"
    white_path = "data/finger_data/session_001/exported_white_2022-07-15-20-29-45.png"

    img = hsi_utils.load_frame(img_path)
    white = hsi_utils.load_sample(white_path)

    hsi_data = HSImage(array=img, wavelengths=None, camera="imec 4x4-VIS-15.7.15.4")
    hsi_data.reconstruct(white=white, dark=None, rho=15, method="flatfield")
    hsi_origin = copy.deepcopy(hsi_data)
    hsi_data.array = utils.normalise_L1(hsi_data.array)

    # Data reformat
    data = hsi_data.array
    shape = data.shape
    data = data.reshape(shape[0], shape[1] * shape[2]).swapaxes(0, 1)

    step = 2 ** 13
    batch = list(np.arange(data.shape[0], step=step))
    batch.append(data.shape[0])

    output = torch.tensor([])
    for i in tqdm(range(len(batch) - 1), ascii=True, ncols=100):
        y = data[batch[i]:batch[i + 1]].to(c.device)
        z = torch.randn(y.shape[0], c.ndim_z).to(c.device)
        y = torch.cat((torch.zeros(y.shape[0], c.ndim_tot - c.ndim_y - c.ndim_z).to(c.device), y), dim=1)
        y = torch.cat((z, y), dim=1).to(c.device)
        pred_x = model(y, rev=True)[0][:, c.target].detach().cpu().clone()
        output = torch.cat((output, pred_x))

    output = output.reshape(shape[1], shape[2])[:, :, None]

    show_inn_combo_norm(hsi_origin, output, mask=False)
