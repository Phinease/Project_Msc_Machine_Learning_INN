from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import norm

import copy
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity

from hvs_hsi_pytorch.utils import utils
from hvs_hsi_pytorch.hs_image import HSImage
from hvs_hsi_pytorch.hsicamera_simulator import HSICameraSimulator


dpi = 300


def plot_original(ax, hsi_image, mask):
    ax.imshow(hsi_image.rgb(mask=mask).cpu())
    ax.axis("off")


def show_hvs_combo(hsi_image, mask, save_name=None):
    fig = plt.figure(figsize=(15, 3))
    ax0 = fig.add_subplot(131)
    ax1 = fig.add_subplot(132)
    ax2 = fig.add_subplot(133)

    plot_original(ax0, hsi_image, mask)
    sto2 = hsi_image.sto2(mask=mask).cpu()
    plot_im_and_dist(ax1, ax2, sto2, 'hvs', 'navy', fig, aim_range=(0, 1))

    plt.tight_layout()
    if save_name:
        fig.savefig('output/hvs/hvs_' + save_name + '.png', dpi=dpi)
    return


def show_inn_combo_norm(hsi_image, hsi_array, mask, aim_std=0.2, aim_range=(-2, 3), save_name=None):
    hsi_array = copy.deepcopy(hsi_array)
    median = hsi_array.median()
    hsi_array[hsi_array < aim_range[0]] = median
    hsi_array[hsi_array > aim_range[1]] = median

    fig = plt.figure(figsize=(15, 3))
    ax0 = fig.add_subplot(131)
    ax1 = fig.add_subplot(132)
    ax2 = fig.add_subplot(133)

    # RGB
    plot_original(ax0, hsi_image, mask=mask)
    
    # Prediction Dist Only
    img = copy.deepcopy(hsi_array)
    if mask:
        img = utils.mask(img, channels=1)
        img[img == 0] = aim_range[0]
    plot_im_and_dist(None, ax2, img, 'origin', 'navy', fig, median=median, aim_range=aim_range)
    
    # Normed Prediction with Dist
    img = copy.deepcopy(hsi_array)
    if mask:
        img = utils.mask(img, channels=1)

    mu, std = norm.fit(img[img != 0])
    img[img != 0] = ((img[img != 0] - mu) / (std / aim_std)) + mu
    img[img == 0] = mu - 0.5
    img[img < mu - 0.5] = mu - 0.5
    img[img > mu + 0.5] = mu + 0.5
    plot_im_and_dist(ax1, ax2, img, 'normed std - ' + str(aim_std), 'green', fig, median=median, aim_range=(mu-0.5, mu+0.5))
    
    plt.tight_layout()
    if save_name:
        fig.savefig('output/norm/norm_' + save_name + '.png', dpi=dpi)
    return


def show_inn_combo_range(hsi_image, hsi_array, mask, aim_std=0.2, aim_range=(-2, 3), save_name=None):
    hsi_array = copy.deepcopy(hsi_array)
    median = hsi_array.median()
    hsi_array[hsi_array < aim_range[0]] = median
    hsi_array[hsi_array > aim_range[1]] = median
    if mask:
        hsi_array = utils.mask(hsi_array, channels=1)
        hsi_array[hsi_array == 0] = aim_range[0]

    fig = plt.figure(figsize=(15, 3))
    ax0 = fig.add_subplot(131)
    ax1 = fig.add_subplot(132)
    ax2 = fig.add_subplot(133)

    # RGB
    plot_original(ax0, hsi_image, mask=mask)

    # Prediction Dist Only
    img = copy.deepcopy(hsi_array)
    plot_im_and_dist(None, ax2, img, 'origin', 'navy', fig, median=median, aim_range=aim_range)

    # Normed Prediction with Dist
    img = copy.deepcopy(hsi_array)
    mu, std = norm.fit(img[img != 0])
    img[img != 0] = ((img[img != 0] - mu) / (std / aim_std)) + mu
    img[img == 0] = aim_range[0]
    img[img < aim_range[0]] = aim_range[0]
    img[img > aim_range[1]] = aim_range[1]
    median = ((median - mu) / (std / aim_std)) + mu
    plot_im_and_dist(ax1, ax2, img, 'normed std - ' + str(aim_std), 'green', fig, median=median, aim_range=aim_range)

    plt.tight_layout()
    if save_name:
        fig.savefig('./output/norm/norm_' + save_name + '.png', dpi=dpi)
    return


def show_inn_combo_origin(hsi_image, hsi_array, mask, aim_range=(0, 1), save_name=None):
    hsi_array = copy.deepcopy(hsi_array)
    median = hsi_array.median()
    hsi_array[hsi_array < aim_range[0]] = median
    hsi_array[hsi_array > aim_range[1]] = median
    if mask:
        hsi_array = utils.mask(hsi_array, channels=1)
        hsi_array[hsi_array == 0] = aim_range[0]

    fig = plt.figure(figsize=(15, 3))
    ax0 = fig.add_subplot(131)
    ax1 = fig.add_subplot(132)
    ax2 = fig.add_subplot(133)

    # RGB
    plot_original(ax0, hsi_image, mask=mask)

    # Prediction Dist Only
    img = copy.deepcopy(hsi_array)
    plot_im_and_dist(ax1, ax2, img, 'origin', 'navy', fig, median=median, aim_range=aim_range)

    plt.tight_layout()
    if save_name:
        fig.savefig('output/origin/origin_' + save_name + '.png', dpi=dpi)
    return


def plot_im_and_dist(ax_main, ax_dist, img, dist_name, color, fig, median=None, aim_range=(0, 1)):
    temp = img[img != aim_range[0]]
    temp = temp[temp != aim_range[1]]
    if median is not None:
        temp = temp[temp != median]
    temp = temp.flatten()[:, None]

    density = KernelDensity(bandwidth=0.01, kernel='gaussian').fit(temp)
    plot_x = np.linspace(aim_range[0], aim_range[1], 500)[:, None]
    log_dens = density.score_samples(plot_x)
    ax_dist.plot(plot_x[:, 0], np.exp(log_dens) / 1.02, c=color, lw=2, linestyle="-", label=dist_name)
    ax_dist.legend()

#     samples = temp[torch.randperm(temp.size(0))[:500]]
#     centers = MeanShift(bandwidth=0.1, n_jobs=-1, min_bin_freq=100).fit(samples).cluster_centers_
#     scores = density.score_samples(centers)
#     mu = centers[np.argmax(scores)].item()
#     ax_dist.axvline(x=mu, c=color)
    
    if ax_main is not None:
        img[img == 0] = img.min()
        img = torch.squeeze(img)
        sto2 = ax_main.imshow(img, cmap='inferno')
        divider = make_axes_locatable(ax_main)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(sto2, cax=cax, orientation="vertical", label="StO$_2$")
        ax_main.axis("off")
    return


def show_simulation_comparison(origin_file_path, simulated_file_path, sensor, idx=(25, 716, 1042, 1389), l1=False):
    # Read origin dataset
    df_origin = pd.read_csv(origin_file_path, skiprows=[0], usecols=list(range(26, 171)))
    wavelengths_origin = torch.tensor([eval(i.split('_')[-1]) * 1e9 for i in df_origin.columns.to_list()])
    df_origin.columns = wavelengths_origin.numpy()

    reflectance_origin = torch.tensor(df_origin.to_numpy()[:, None], dtype=torch.float32)
    reflectance_origin = torch.swapaxes(reflectance_origin, 0, 2)
    reflectance_origin = torch.swapaxes(reflectance_origin, 1, 2)
    hsi_image_origin = HSImage(reflectance_origin, wavelengths=wavelengths_origin, camera=None)

    # Read simulated dataset
    df_simulation = pd.read_csv(simulated_file_path, usecols=list(range(24, 40)))
    df_simulation = df_simulation.iloc[:1500]
    wavelengths_simulation = torch.tensor([eval(i.split('_')[-1]) for i in df_simulation.columns.to_list()])
    df_simulation.columns = wavelengths_simulation.numpy()

    reflectance_simulation = torch.tensor(df_simulation.to_numpy()[:, None], dtype=torch.float32)
    reflectance_simulation = torch.swapaxes(reflectance_simulation, 0, 2)
    reflectance_simulation = torch.swapaxes(reflectance_simulation, 1, 2)
    hsi_image_simulation = HSImage(reflectance_simulation, wavelengths=wavelengths_simulation, camera=None)

    # Simulation by HSICameraSimulator
    imec_camera = HSICameraSimulator(camera=sensor)
    hsi_image_imec = imec_camera(hsi_image_origin)

    if idx is None:
        idx = torch.randint(0, 1000, (4,))

    if l1:
        hsi_image_origin.array = utils.normalise_L1(hsi_image_origin.array)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    for i, ax in enumerate(axs.flatten()):
        ax.plot(hsi_image_origin.wavelengths, hsi_image_origin.array[:, idx[i], :], label='Original')
        ax.plot(hsi_image_imec.wavelengths, hsi_image_imec.array[:, idx[i], :], label='Imec')
        ax.plot(hsi_image_simulation.wavelengths, hsi_image_simulation.array[:, idx[i], :], label='Simulation Pipeline')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Reflectance')
        ax.legend()

    plt.tight_layout()
    return


def plot_valid_simple(true_y, pred_y, label):
    plt.figure()
    fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(9, 6))
    axs.set_title(label)
    axs.scatter(true_y, pred_y, alpha=0.2)
    axs.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), c='red')
    axs.set_ylim([-0.8, 1.8])
    plt.show()


def plot_valid_each(ax, true_y, pred_y, title, label, aim_range):
    ax.set_title(title)
    ax.scatter(true_y, pred_y, alpha=0.2, label=label)
    ax.plot(np.linspace(true_y.min(), true_y.max(), 100), np.linspace(true_y.min(), true_y.max(), 100), c='red')
    ax.set_ylim(aim_range)
    ax.legend()
