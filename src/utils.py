import os
import numpy as np
from natsort import natsorted
import warnings

import torch
import pandas as pd

from tifffile import TiffFile
import imageio
import cv2

from hvs_hsi_pytorch.imec_operators import ImecFilterOperator, HypercubeAcquisitionImecOperator
import hvs_hsi_pytorch
from hvs_hsi_pytorch.hs_image import HSImage

ROOT_DIR = list(hvs_hsi_pytorch.__path__)[0]
wavelengths_nuance = np.loadtxt(os.path.join(ROOT_DIR, 'resources/wavelengths_odsi_nuance.txt'))
wavelengths_specim = np.loadtxt(os.path.join(ROOT_DIR, 'resources/wavelengths_odsi_specim.txt'))


def read_stiff(filename: str, silent=False, rgb_only=False):
    """

    :param filename:    filename of the spectral tiff to read.
    :return:            Tuple[spim, wavelengths, rgb, metadata], where
                        spim: spectral image cube of form [height, width, bands],
                        wavelengths: the center wavelengths of the bands,
                        rgb: a color render of the spectral image [height, width, channels] or None
                        metadata: a free-form metadata string stored in the image, or an empty string
    """
    TIFFTAG_WAVELENGTHS = 65000
    TIFFTAG_METADATA = 65111
    spim = None
    wavelengths = None
    rgb = None
    metadata = None

    first_band_page = 0
    with TiffFile(filename) as tiff:
        # The RGB image is optional, the first band image maybe on the first page:
        first_band_page = 0
        if tiff.pages[first_band_page].ndim == 3:
            rgb = tiff.pages[0].asarray()
            # Ok, the first band image is on the second page
            first_band_page = first_band_page + 1

        multiple_wavelength_lists = False
        multiple_metadata_fields = False
        for band_page in range(first_band_page, len(tiff.pages)):
            # The wavelength list is supposed to be on the first band image.
            # The older write_tiff writes it on all pages, though, so make
            # a note of it.
            tag = tiff.pages[band_page].tags.get(TIFFTAG_WAVELENGTHS)
            tag_value = tag.value if tag else tuple()
            if tag_value:
                if wavelengths is None:
                    wavelengths = tag_value
                elif wavelengths == tag_value:
                    multiple_wavelength_lists = True
                elif wavelengths != tag_value:
                    # Well, the image is just broken then?
                    raise RuntimeError(f'Spectral-Tiff "{filename}" contains multiple differing wavelength lists!')

            # The metadata string, like the wavelength list, is supposed to be
            # on the first band image. The older write_tiff wrote it on all
            # pages, too. Make a note of it.
            tag = tiff.pages[band_page].tags.get(TIFFTAG_METADATA)
            tag_value = tag.value if tag else ''
            if tag_value:
                if metadata is None:
                    metadata = tag_value
                elif metadata == tag_value:
                    multiple_metadata_fields = True
                elif metadata != tag_value:
                    # Well, for some reason there are multiple metadata fields
                    # with varying content. This version of the function does
                    # not care for such fancyness.
                    raise RuntimeError(f'Spectral-Tiff "{filename}" contains multiple differing metadata fields!')

        # The metadata is stored in an ASCII string. It may contain back-slashed
        # hex sequences (unicode codepoints presented as ASCII text). Convert
        # ASCII string back to bytes and decode as unicode sequence.
        if metadata:
            metadata = metadata.encode('ascii').decode('unicode-escape')
        else:
            metadata = ''

        # Some of the early images may have errorneus metadata string.
        # Attempt to fix it:
        if metadata and metadata[0] == "'" and metadata[-1] == "'":
            while metadata[0] == "'":
                metadata = metadata[1:]
            while metadata[-1] == "'":
                metadata = metadata[:-1]
            if '\\n' in metadata:
                metadata = metadata.replace('\\n', '\n')

        # Generate a fake wavelength list, if the spectral tiff has managed to
        # lose its own wavelength list.
        if not wavelengths:
            wavelengths = range(0, len(tiff.pages) - 1 if rgb is not None else len(tiff.pages))

        if multiple_wavelength_lists and not silent:
            warnings.warn(f'Spectral-Tiff "{filename}" contains duplicated wavelength lists!')
        if multiple_metadata_fields and not silent:
            warnings.warn(f'Spectral-Tiff "{filename}" contains duplicated metadata fields!')

        if not rgb_only:
            spim = tiff.asarray(key=range(first_band_page, len(tiff.pages)))
            spim = np.transpose(spim, (1, 2, 0))
        else:
            spim = None

        # Make sure the wavelengths are in an ascending order:
        if wavelengths[0] > wavelengths[-1]:
            spim = spim[:, :, ::-1] if spim is not None else None
            wavelengths = wavelengths[::-1]

    # Convert uint16 cube back to float32 cube
    if spim is not None and spim.dtype == 'uint16':
        spim = spim.astype('float32') / (2 ** 16 - 1)

    return spim, np.array(wavelengths), rgb, metadata


def parse_spectrodata(path):
    with open(path, encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    start = [lines.index(k) for k in lines if 'Begin' in k]
    assert len(start) == 1
    start = start[0]

    wavelengths = []
    intensities = []

    for row in lines[start + 1:]:
        wavelengths.append(float(row.split('\t')[0]))
        intensities.append(float(row.split('\t')[1].replace('\n', '')))
    return pd.DataFrame({'wavelength (nm)': wavelengths, 'intensity': intensities})


def demosaic(array, mosaic=(4, 4)):
    output = np.array(
        [
            cv2.warpAffine(
                src=array[i:array.shape[0]:mosaic[0], j:array.shape[1]:mosaic[1]],
                dsize=(array.shape[1], array.shape[0]),
                M=np.float32([[mosaic[1], 0, j], [0, mosaic[0], i]]),
                flags=cv2.INTER_LINEAR
            )
            for i in range(mosaic[0]) for j in range(mosaic[1])
        ]
    )
    return output


def get_camera_response(hypercube, sensor, precision=torch.float32):
    a_filter = ImecFilterOperator(sensor)
    output_array = a_filter(hypercube.array, hypercube.wavelengths).to(precision)
    output = HSImage(array=output_array, wavelengths=hypercube.wavelengths)
    output = HypercubeAcquisitionImecOperator(sensor, precision=precision)(output.array, output.wavelengths)
    return output


def get_white_image(path_white_images, num_sensor, light):
    white_path = [
        os.path.join(path_white_images, num_sensor, light, k)
        for k in os.listdir(os.path.join(path_white_images, num_sensor, light))
        if '.png' in k
    ]
    assert len(white_path) == 1

    white = imageio.imread(white_path[0])
    return white


def get_dark_image(path_dark_images, num_sensor):
    dark_path = natsorted(
        [
            os.path.join(path_dark_images, num_sensor, k)
            for k in os.listdir(os.path.join(path_dark_images, num_sensor))
            if '.png' in k
        ]
    )
    dark = np.stack([imageio.imread(k) for k in dark_path], 0).mean(0)
    return dark


def get_mask(demosaiced_img, threshold=0.05):
    normed_img = demosaiced_img - demosaiced_img.min(1).min(1)[:, None, None]
    normed_img /= normed_img.max(1).max(1)[:, None, None]
    mask = np.all(normed_img > threshold, 0)
    return mask


def load_spectrum(path_spectrum, light, wavelengths_ref):
    spectrum = parse_spectrodata(path_spectrum)
    intensities_light = spectrum['intensity'].values
    wavelengths_light = spectrum['wavelength (nm)'].values

    if not wavelengths_ref is None:
        intensity_light_inputwave = np.interp(
            x=wavelengths_ref,
            xp=wavelengths_light,
            fp=intensities_light
        )
        return torch.tensor(intensity_light_inputwave)
    else:
        return torch.tensor(intensities_light), torch.tensor(wavelengths_light)
