import warnings

import hydra
import torch
from hydra.utils import instantiate
import logging

from src.datasets.data_utils import get_dataloaders
from src.trainer import Inferencer
from src.utils.io_utils import ROOT_PATH
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from pathlib import Path
import time

from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    """
    Main script for inference. Instantiates the model, metrics, and
    dataloaders. Runs Inferencer to calculate metrics and (or)
    save predictions.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.inferencer.seed)

    if config.inferencer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.inferencer.device

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device)
    if config.writer.get("run_name") is not None:
        project_config = OmegaConf.to_container(config)
        logger = setup_saving_and_logging(config)
        writer = instantiate(config.writer, logger, project_config)
    else:
        logger = logging.getLogger("inference")
        writer = None

    model_name = config.model_name
    model_weights = {
        "LeADMM-5-pre-post-inv": ("model_weights_8m_le_admm_5_pre_post_inv.pth", "le_admm_5_pre_post_inv"),
        "LeADMM-5-post-inv": ("model_weights_le_admm_5_post_inv_corr.pth", "le_admm_5_post_inv"),
        "LeADMM-5-pre-inv": ("model_weights_le_admm_5_pre_inv_corr.pth", "le_admm_5_pre_inv"),
        "Unrolled_ADMM-20": ("model_weights_unrolled_admm_20_corr.pth", "unrolled_admm_20"),
    }

    if model_name is None:
        raise RuntimeError("No model name provided")
    else:
        if model_name == "ADMM-100":
            weights = None
            yaml_path = "admm_100"
        else:
            if model_name not in model_weights:
                raise RuntimeError("Unknown model name")
            weights = model_weights[model_name][0]
            yaml_path = model_weights[model_name][1]
    config_path = ROOT_PATH / "src" / "configs" / "model" / f"{yaml_path}.yaml"
    assert config_path.is_file(), "config file not found"
    config_model = OmegaConf.load(config_path)
    model = instantiate(config_model).to(device)
    if weights is not None:
        check = hf_hub_download(repo_id="maxsnytsarev/lensless_camera", filename=weights,
                                repo_type="model")
        checkpoint = torch.load(check, map_location=device, weights_only=False)
        state_dict = checkpoint["state_dict"]
        model.load_state_dict(state_dict)
        print("Loaded model weights")
    else:
        print("No weights are loaded. Using initialized model.")
    print("Model is initialized")

    # get metrics
    metrics = instantiate(config.metrics)

    # save_path for model predictions
    _path = config.inferencer.get("save_path")
    if _path is None:
        print("No save path provided. Using default save path. 'inference_data/saved'")
        _path = Path("inference_data") / "saved"
    save_path = ROOT_PATH / _path
    save_path.mkdir(exist_ok=True, parents=True)

    inferencer = Inferencer(
        model=model,
        config=config,
        device=device,
        dataloaders=dataloaders,
        batch_transforms=batch_transforms,
        save_path=save_path,
        metrics=metrics,
        writer=writer,
        skip_model_load=True,
    )
    if config.inferencer.get("eval_time") == True:
        dataloader = next(iter(dataloaders.values()))
        batch = next(iter(dataloader))
        batch = inferencer.move_batch_to_device(batch)
        warmup = 3
        runs = 10
        batch_size = batch["lensless"].shape[0]
        print("Started time evaluation")
        model.eval()
        with torch.no_grad():
            for i in range(warmup):
                model(**batch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            for i in range(runs):
                model(**batch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        total_time = time.perf_counter() - start
        time_ms = 1000 * total_time / (runs * batch_size)
        print(f"Total time: {time_ms} ms/image")

        if writer is not None:
            writer.set_step(0, "speed")
            writer.add_scalar("Average time (ms)", time_ms)

    logs = inferencer.run_inference()
    for part in logs.keys():
        for key, value in logs[part].items():
            full_key = part + "_" + key
            print(f"    {full_key:15s}: {value}")


if __name__ == "__main__":
    main()
