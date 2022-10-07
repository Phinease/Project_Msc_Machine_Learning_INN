import torch
import torch.utils.data as data_utils

import numpy as np


class McDataset(data_utils.Dataset):
    def __init__(self, data_mc, c):
        parameters = c.parameters
        n_spectrum_start = c.n_spectrum_start

        selected_columns = []
        for param in parameters:
            selected_columns += [i for i in data_mc.columns if i.find(param) != -1]
            
        c.targets = selected_columns

        self._y = torch.tensor(data_mc.iloc[:, n_spectrum_start:].values.astype(np.float32), dtype=torch.float32)
        self._x = torch.tensor(data_mc[selected_columns].values.astype(np.float32), dtype=torch.float32)

    def __len__(self):
        return len(self._y)

    def __getitem__(self, index):
        return self._x[index], self._y[index]

    def get_y_dim(self):
        return self._y.shape[1]

    def get_x_dim(self):
        return self._x.shape[1]
