from torch import nn
from src.model.DRUNet import DRUNet
from src.model.Le_ADMM import Le_ADMM
import torch.nn.functional as F

def pad_(x):
    b, c, h, w = x.shape
    h_m = h
    w_m = w
    if h_m % 8 != 0:
        h_m += 8 - h_m % 8
    if w_m % 8 != 0:
        w_m += 8 - w_m % 8
    h_pad = h_m - h
    w_pad = w_m - w
    x = F.pad(x, (0, w_pad, 0, h_pad))
    return x

def crop(x, h_true, w_true):
    return x[:, :, :h_true, :w_true]

class FullModel(nn.Module):
    def __init__(self, k, input_channels=3, mode=(False, False), trainable=False, h=380, w=507):
        super().__init__()
        self.mode = mode
        self.k = k
        self.h = h
        self.w = w
        self.input_channels = input_channels
        self.parts = nn.ModuleList()
        if self.mode[0]:
            self.pre_process = DRUNet(input_channels=input_channels, channels=(32, 64, 128, 256))
        else:
            self.pre_process = None
        self.image_inverse = Le_ADMM(h=h, w=w, trainable=trainable, k=k)
        if self.mode[1]:
            self.post_process = DRUNet(input_channels=input_channels, channels=(32, 64, 128, 256))
        else:
            self.post_process = None

    def forward(self, lensless, psf, **batch):
        if self.mode[0]:
            lensless = pad_(lensless)
            lensless = self.pre_process(lensless)
            lensless = crop(lensless, self.h, self.w)
        reconstructed = self.image_inverse(lensless, psf)
        if self.mode[1]:
            reconstructed = pad_(reconstructed)
            reconstructed = self.post_process(reconstructed)
            reconstructed = crop(reconstructed, self.h, self.w)
        return {"reconstructed": reconstructed}

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info