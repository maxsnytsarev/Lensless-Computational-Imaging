import warnings
import lpips
import hydra
import torch
from hydra.utils import instantiate

from src.datasets.data_utils import get_dataloaders
from src.trainer import Inferencer
from src.utils.init_utils import set_random_seed
from src.utils.io_utils import ROOT_PATH
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
import logging
from src.utils.init_utils import set_random_seed, setup_saving_and_logging
from src.metrics.image_metrics import *
import torchvision.io as tv_io
import torch.nn.functional as F
from tqdm.auto import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="get_metrics")
def main(config):
    """
    Main script for inference. Instantiates the model, metrics, and
    dataloaders. Runs Inferencer to calculate metrics and (or)
    save predictions.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.metrics_info.seed)

    if config.metrics_info.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.metrics_info.device

    # setup data_loader instances
    # batch_transforms should be put on device
    project_config = OmegaConf.to_container(config)
    if config.writer.get("run_name") is not None:
        project_config = OmegaConf.to_container(config)
        logger = setup_saving_and_logging(config)
        writer = instantiate(config.writer, logger, project_config)
    else:
        logger = logging.getLogger("get_metrics")
        writer = None
    # build model architecture, then print to console

    if config.metrics_info.get("metrics_path") is not None:
        path_to_metrics = ROOT_PATH / config.metrics_info.metrics_path
    else:
        raise RuntimeError("No path to real/recovered data for metrics calculation provided")
    if not path_to_metrics.is_dir():
        raise FileNotFoundError(f"{path_to_metrics} not found")

    batch_lensed = []
    batch_reconstructed = []
    all_id_dir = list(path_to_metrics.iterdir())
    for id_dir in all_id_dir:
        cur_id = id_dir.name
        cur_path = id_dir
        if not id_dir.is_dir():
            raise RuntimeError(f"{id_dir} is not a directory")
        cur_lensed = cur_path / f"lensed_roi_{cur_id}.png"
        cur_reconstructed = cur_path / f"reconstructed_roi_{cur_id}.png"
        if cur_lensed.is_file() and cur_reconstructed.is_file():
            lensed_tensor = tv_io.read_image(cur_lensed)
            reconstructed_tensor = tv_io.read_image(cur_reconstructed)
            batch_lensed.append(lensed_tensor)
            batch_reconstructed.append(reconstructed_tensor)
        else:
            raise RuntimeError(f"{cur_lensed} or {cur_reconstructed} is missing")
    # save_path for model predictions

    def pad_(elem, H, W):
        c, h, w = elem.shape
        elem = F.pad(elem, pad=(0, W - w, 0, H - h), mode="constant", value=0)
        return elem

    cnt = len(batch_lensed)
    mse = 0
    lpips_sum = 0
    psnr = 0
    ssim = 0

    def pre_process(img):
        img = img.clamp(0, 1)
        img = img.detach().cpu()
        img = img.permute(0, 2, 3, 1).numpy()
        return img

    with torch.no_grad():
        lpips_metric = lpips.LPIPS(net="vgg").to(device)
        for i in tqdm(range(cnt)):
            cur_lensed = batch_lensed[i] / 255.0
            cur_reconstructed = batch_reconstructed[i] / 255.0
            cur_lensed = cur_lensed.unsqueeze(0).to(device)
            cur_reconstructed = cur_reconstructed.unsqueeze(0).to(device)

            cur_lensed_np = pre_process(cur_lensed)
            cur_reconstructed_np = pre_process(cur_reconstructed)

            mse += nn.MSELoss()(cur_lensed, cur_reconstructed).item()

            lpips_sum += lpips_metric(cur_lensed, cur_reconstructed).item()

            psnr += psnr_metric(cur_lensed_np[0], cur_reconstructed_np[0], data_range=1.0)

            ssim += ssim_metric(2 * cur_lensed_np[0]-1, 2*cur_reconstructed_np[0]-1, data_range=2.0, channel_axis=-1)

    mse /= cnt
    lpips_sum /= cnt
    psnr /= cnt
    ssim /= cnt

    print(f"Mean MSE: {mse}")
    print(f"Mean LPIPS: {lpips_sum}")
    print(f"Mean PSNR: {psnr}")
    print(f"Mean SSIM: {ssim}")

    if writer is not None:
        writer.set_step(0, "metrics")
        writer.add_scalar("MSE", mse)
        writer.add_scalar("LPIPS", lpips_sum)
        writer.add_scalar("PSNR", psnr)
        writer.add_scalar("SSIM", ssim)


if __name__ == "__main__":
    main()
