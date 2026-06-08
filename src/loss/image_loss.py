import torch
from torch import nn
import lpips
from lensless_helpers.preprocessor import get_roi_bchw

class ImageLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_lpips = lpips.LPIPS(net="vgg")
        self.mse = nn.MSELoss()

    def forward(self, lensed: torch.Tensor, reconstructed: torch.Tensor, **batch):
        reconstructed = get_roi_bchw(reconstructed)
        lensed = get_roi_bchw(lensed)
        lpips_loss = self.loss_lpips(2*reconstructed-1, 2*lensed-1).mean()
        mse = self.mse(reconstructed, lensed)
        return {"loss": mse + lpips_loss, "loss_mse": mse, "loss_lpips": lpips_loss}
