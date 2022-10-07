import torch
import numpy as np

# def loss_mmd_multiscale(x, y, c):
#     xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
#
#     rx = (xx.diag().unsqueeze(0).expand_as(xx))
#     ry = (yy.diag().unsqueeze(0).expand_as(yy))
#
#     dxx = rx.t() + rx - 2. * xx
#     dyy = ry.t() + ry - 2. * yy
#     dxy = rx.t() + ry - 2. * zz
#
#     out_xx, out_yy, out_xy = (torch.zeros(xx.shape).to(c.device),
#                               torch.zeros(xx.shape).to(c.device),
#                               torch.zeros(xx.shape).to(c.device))
#
#     for C, a in [(0.2, 2), (1.5, 2), (3.0, 2)]: # [0.05, 0.2, 0.9]
#         out_xx += C ** a * ((C + dxx) / a) ** -a
#         out_yy += C ** a * ((C + dyy) / a) ** -a
#         out_xy += C ** a * ((C + dxy) / a) ** -a
#
#     return torch.mean(out_xx + out_yy - 2. * out_xy)


def loss_mse(x, y, c):
    return torch.nn.MSELoss()(x, y)


def loss_huber(x, y, c):
    return torch.nn.HuberLoss()(x, y)


def loss_mae(x, y, c):
    return torch.nn.L1Loss()(x, y)


def loss_mmd_multiscale(x, y, widths_exponents, c):
    xx, yy, xy = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = torch.clamp(rx.t() + rx - 2. * xx, 0, np.inf)
    dyy = torch.clamp(ry.t() + ry - 2. * yy, 0, np.inf)
    dxy = torch.clamp(rx.t() + ry - 2. * xy, 0, np.inf)

    XX, YY, XY = (torch.zeros(xx.shape).to(c.device),
                  torch.zeros(xx.shape).to(c.device),
                  torch.zeros(xx.shape).to(c.device))

    for C, a in widths_exponents:
        XX += C ** a * ((C + dxx) / a) ** -a
        YY += C ** a * ((C + dyy) / a) ** -a
        XY += C ** a * ((C + dxy) / a) ** -a

    return torch.mean(XX + YY - 2. * XY)


def forward_mmd(y0, y1, c):
    return loss_mmd_multiscale(y0, y1, c.mmd_forw_kernels, c)


def backward_mmd(x0, x1, c):
    return loss_mmd_multiscale(x0, x1, c.mmd_back_kernels, c)
