import lpips
import numpy as np
import torch
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from lensless_helpers.preprocessor import get_roi_bchw
from src.metrics.base_metric import BaseMetric


def after_roi(lensed, recon):
    reconstructed = get_roi_bchw(recon)
    lensed = get_roi_bchw(lensed)
    return lensed, reconstructed


def pre_process(img):
    img = img.clamp(0, 1)
    img = img.detach().cpu()
    img = img.permute(0, 2, 3, 1).numpy()
    return img


class MSEMetric(BaseMetric):
    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = nn.MSELoss().to(device)

    def __call__(self, lensed, reconstructed, **kwargs):
        lensed, reconstructed = after_roi(lensed, reconstructed)
        return self.metric(lensed, reconstructed).mean()


class PSNRMetric(BaseMetric):
    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric = psnr

    def __call__(self, lensed, reconstructed, **kwargs):
        lensed, reconstructed = after_roi(lensed, reconstructed)
        lensed, reconstructed = pre_process(lensed), pre_process(reconstructed)
        metric_value = [
            self.metric(lensed[i], reconstructed[i], data_range=1.0)
            for i in range(len(lensed))
        ]
        return np.mean(metric_value)


class SSIMMetric(BaseMetric):
    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric = ssim

    def __call__(self, lensed, reconstructed, **kwargs):
        lensed, reconstructed = after_roi(lensed, reconstructed)
        lensed, reconstructed = pre_process(lensed), pre_process(reconstructed)
        metric_value = [
            self.metric(lensed[i], reconstructed[i], data_range=1.0, channel_axis=-1)
            for i in range(len(lensed))
        ]
        return np.mean(metric_value)


class LPIPSMetric(BaseMetric):
    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.metric = lpips.LPIPS(net="vgg").to(device)

    def __call__(self, lensed, reconstructed, **kwargs):
        lensed, reconstructed = after_roi(lensed, reconstructed)
        lensed = lensed.to(self.device)
        reconstructed = reconstructed.to(self.device)
        return self.metric(2 * lensed - 1, 2 * reconstructed - 1).mean()
