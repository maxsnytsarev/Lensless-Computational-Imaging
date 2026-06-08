import torch
import torch.nn.functional as F
from torch import nn


def psi(x):
    dx = torch.roll(x, shifts=1, dims=2) - x
    dy = torch.roll(x, shifts=1, dims=3) - x
    return torch.cat([dx, dy], dim=1)


def psiT(q):
    C = q.shape[1]
    qx = q[:, : C // 2, :, :]
    qy = q[:, C // 2 :, :, :]
    dxT = torch.roll(qx, shifts=-1, dims=2) - qx
    dyT = torch.roll(qy, shifts=-1, dims=3) - qy
    return dxT + dyT


def soft_thresholding(ind, x):
    return torch.sign(x) * torch.where(
        torch.abs(x) - ind > 0, torch.abs(x) - ind, torch.zeros_like(x)
    )


def Hx(H_fft, x):
    X = torch.fft.fft2(x, dim=(-2, -1))
    Y = H_fft * X
    return torch.fft.ifft2(Y, dim=(-2, -1)).real


def HTx(H_fft, x):
    X = torch.fft.fft2(x, dim=(-2, -1))
    H_conj = torch.conj(H_fft)
    Y = H_conj * X
    return torch.fft.ifft2(Y, dim=(-2, -1)).real


def center_crop(x, H_pad, W_pad, h, w):
    h_ = (H_pad - h) // 2
    w_ = (W_pad - w) // 2
    return x[:, :, h_ : h_ + h, w_ : w_ + w]


def center_pad(x, H_pad, W_pad, h, w):
    pad_top = (H_pad - h) // 2
    pad_bottom = H_pad - h - pad_top
    pad_left = (W_pad - w) // 2
    pad_right = W_pad - w - pad_left
    padding = (pad_left, pad_right, pad_top, pad_bottom)
    padded_x = F.pad(x, padding, mode="constant", value=0)
    return padded_x


class ADMM_Layer(nn.Module):
    def __init__(self, h, w, trainable):
        super().__init__()
        self.trainable = trainable
        self.h = h
        self.w = w
        H_pad = 2 * h
        W_pad = 2 * w
        self.H_pad = H_pad
        self.W_pad = W_pad
        CTC = torch.zeros((H_pad, W_pad))
        h_ = (H_pad - h) // 2
        W_ = (W_pad - w) // 2
        CTC[h_ : h_ + h, W_ : W_ + w] = torch.ones((h, w))
        self.register_buffer("CTC", CTC)
        dx = torch.zeros((1, 2))
        dx[0, 0] = -1
        dx[0, 1] = 1
        dy = torch.zeros((2, 1))
        dy[0, 0] = -1
        dy[1, 0] = 1
        dx_fft = (
            torch.fft.fft2(dx, dim=(-2, -1), s=(H_pad, W_pad)).unsqueeze(0).unsqueeze(0)
        )
        self.register_buffer("dx_fft", dx_fft)
        dy_fft = (
            torch.fft.fft2(dy, dim=(-2, -1), s=(H_pad, W_pad)).unsqueeze(0).unsqueeze(0)
        )
        self.register_buffer("dy_fft", dy_fft)
        if trainable:
            self.mu1 = nn.Parameter(torch.tensor(1e-4))
            self.mu2 = nn.Parameter(torch.tensor(1e-4))
            self.mu3 = nn.Parameter(torch.tensor(1e-4))
            self.tau = nn.Parameter(torch.tensor(2e-4))
        else:
            self.mu1 = 1e-4
            self.mu2 = 1e-4
            self.mu3 = 1e-4
            self.tau = 2e-4

    def forward(self, a1, a2, a3, x, b, H_fft):
        if self.trainable:
            u = soft_thresholding(self.tau, psi(x) + a2 / self.mu2)
        else:
            u = soft_thresholding(self.tau / self.mu2, psi(x) + a2 / self.mu2)
        v = (
            a1
            + self.mu1 * Hx(H_fft, x)
            + center_pad(b, self.H_pad, self.W_pad, self.h, self.w)
        )
        v = v / (self.CTC + self.mu1).unsqueeze(0).unsqueeze(0)
        w = torch.where(a3 / self.mu3 + x > 0, a3 / self.mu3 + x, torch.zeros_like(x))
        r = (
            (self.mu3 * w - a3)
            + psiT(self.mu2 * u - a2)
            + HTx(H_fft, self.mu1 * v - a1)
        )
        denom = (
            self.mu1 * torch.abs(H_fft) ** 2
            + self.mu2 * (torch.abs(self.dx_fft) ** 2 + torch.abs(self.dy_fft) ** 2)
            + self.mu3
        )
        x_new = torch.fft.ifft2(
            torch.fft.fft2(r, dim=(-2, -1)) / denom, dim=(-2, -1)
        ).real
        a1_new = a1 + self.mu1 * (Hx(H_fft, x_new) - v)
        a2_new = a2 + self.mu2 * (psi(x_new) - u)
        a3_new = a3 + self.mu3 * (x_new - w)
        return a1_new, a2_new, a3_new, x_new


class Le_ADMM(nn.Module):
    def __init__(self, h, w, trainable, k):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(k):
            if trainable:
                self.layers.append(ADMM_Layer(h, w, True))
            else:
                self.layers.append(ADMM_Layer(h, w, False))

    def forward(self, b, psf):
        batch, c, h, w = b.shape
        H_pad = 2 * h
        W_pad = 2 * w
        psf = center_pad(psf, H_pad, W_pad, h, w)
        psf = torch.fft.ifftshift(psf, dim=(-2, -1))
        H_fft = torch.fft.fft2(psf, dim=(-2, -1))
        x = torch.zeros(batch, c, H_pad, W_pad, device=b.device)
        a1 = torch.zeros(batch, c, H_pad, W_pad, device=x.device)
        a2 = torch.zeros(batch, 2 * c, H_pad, W_pad, device=x.device)
        a3 = torch.zeros(batch, c, H_pad, W_pad, device=x.device)
        for layer in self.layers:
            a1, a2, a3, x = layer(a1, a2, a3, x, b, H_fft)
        return center_crop(x, H_pad, W_pad, h, w)
