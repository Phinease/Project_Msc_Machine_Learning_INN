from FrEIA.modules import *
from src.loss_funs import *

# Initialization
device = 'cpu'
# device = 'mps' if torch.has_mps else device
device = 'cuda' if torch.cuda.is_available() else device

# Model configuration
n_layers, n_neurons = 10, 128

# Training process parameters
n_epochs = 300
n_its_per_epoch = 200
batch_size = 500

valid_percent = 0.1
valid_batch_size = 1000
init_scale = 0.01
early_stop_mse = 0.05

# Target tissu parameters
parameters = None

# Problem dimensionality
n_spectrum_start = 24
ndim_tot, ndim_x, ndim_y, ndim_z = None, None, None, 20

# Macro settings
verbose = 1
device = device

# Noise parameters
y_noise_scale = 1e-5
z_noise_scale = 1e-2
zeros_noise_scale = 1e-2

# Learning rates and optimizer
lr = 1e-3
l2_reg = 1e-4
betas = (0.9, 0.95)
eps = 1e-6
optimizer = None

# Loss weights
lambda_fit, lambda_latent, lambda_backward, lambda_reconstruct = 1., 50., 100., 1.
mmd_forw_kernels = [(0.2, 2), (1.5, 2), (3.0, 2)]
mmd_back_kernels = [(0.2, 0.1), (0.2, 0.5), (0.2, 2)]

# Loss forward functions
loss_latent = forward_mmd
loss_fit = loss_mse

# Loss backward function
loss_backward = loss_mse
loss_reconstruct = backward_mmd

# Metrics function
target = 0
targets = None
metric_mae = torch.nn.L1Loss()

# Block type
block = RNVPCouplingBlock
# block = RNVPCouplingBlock

# Others
parameters = ['layer0_sao2', 'layer0.1_vhb', 'layer0.2_a_mie']
logger = None