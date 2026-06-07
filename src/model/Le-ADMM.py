import torch
from torch import nn


def soft_thresholding(ind, x):
    return torch.sign(x) * torch.where(
        torch.abs(x) - ind > 0, torch.abs(x) - ind, torch.zeros_like(x)
    )


class ADMM_Layer(nn.Module):
    def __init__(self, C, H, psi, trainable):
        super().__init__()
        self.register_buffer("C", C)
        self.register_buffer("H", H)
        self.register_buffer("psi", psi)
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

    def forward(self, a1, a2, a3, x, b):
        u = soft_thresholding(self.tau, self.psi @ x + a2 / self.mu2)
        k = self.C.shape[-1]
        v = torch.linalg.solve(
            self.C.T @ self.C + self.mu1 * torch.eye(k, device=x.device),
            a1 + self.mu1 * self.H @ x + self.C.T @ b,
        )
        w = torch.where(a3 / self.mu3 + x > 0, a3 / self.mu3 + x, torch.zeros_like(x))
        m = self.psi.shape[-1]
        r = (
            (self.mu3 * w - a3)
            + self.psi.T @ (self.mu2 * u - a2)
            + self.H.T @ (self.mu1 * v - a1)
        )
        x_new = torch.linalg.solve(
            self.mu1 * self.H.T @ self.H
            + self.mu2 * self.psi.T @ self.psi
            + self.mu3 * torch.eye(m, device=x.device),
            r,
        )
        a1_new = a1 + self.mu1 * (self.H @ x_new - v)
        a2_new = a2 + self.mu2 * (self.psi @ x_new - u)
        a3_new = a3 + self.mu3 * (x_new - w)
        return a1_new, a2_new, a3_new, x_new


class Le_ADMM(nn.Module):
    def __init__(self, C, H, psi, trainable):
        super().__init__()
        self.register_buffer("C", C)
        self.register_buffer("H", H)
        self.register_buffer("psi", psi)
        self.layers = nn.ModuleList()
        for i in range(5):
            if trainable:
                self.layers.append(ADMM_Layer(C, H, psi, True))
            else:
                self.layers.append(ADMM_Layer(C, H, psi, False))

    def forward(self, b):
        b = b.T
        x = torch.zeros(self.H.shape[1], b.shape[1], device=b.device)
        a1 = torch.zeros(self.H.shape[0], x.shape[1], device=x.device)
        a2 = torch.zeros(self.psi.shape[0], x.shape[1], device=x.device)
        a3 = torch.zeros(x.shape[0], x.shape[1], device=x.device)
        for layer in self.layers:
            a1, a2, a3, x = layer(a1, a2, a3, x, b)
        return x.T
