import torch
from tqdm.auto import tqdm

from pathlib import Path
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
import torch.nn.functional as F
import torchvision.utils
from lensless_helpers.preprocessor import get_roi_bchw

class Inferencer(BaseTrainer):
    """
    Inferencer (Like Trainer but for Inference) class

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model on the dataset, save predictions, etc.
    """

    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        save_path,
        metrics=None,
        batch_transforms=None,
        skip_model_load=False,
    ):
        """
        Initialize the Inferencer.

        Args:
            model (nn.Module): PyTorch model.
            config (DictConfig): run config containing inferencer config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            save_path (str): path to save model predictions and other
                information.
            metrics (dict): dict with the definition of metrics for
                inference (metrics[inference]). Each metric is an instance
                of src.metrics.BaseMetric.
            batch_transforms (dict[nn.Module] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
            skip_model_load (bool): if False, require the user to set
                pre-trained checkpoint path. Set this argument to True if
                the model desirable weights are defined outside of the
                Inferencer Class.
        """
        assert (
            skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = self.config.inferencer

        self.device = device

        self.model = model
        self.batch_transforms = batch_transforms

        # define dataloaders
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}

        # path definition

        self.save_path = save_path

        # define metrics
        self.metrics = metrics
        if self.metrics is not None:
            self.evaluation_metrics = MetricTracker(
                *[m.name for m in self.metrics["inference"]],
                writer=None,
            )
        else:
            self.evaluation_metrics = None

        if not skip_model_load:
            # init model
            self._from_pretrained(config.inferencer.get("from_pretrained"))

    def run_inference(self):
        """
        Run inference on each partition.

        Returns:
            part_logs (dict): part_logs[part_name] contains logs
                for the part_name partition.
        """
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            logs = self._inference_part(part, dataloader)
            part_logs[part] = logs
        return part_logs

    def process_batch(self, batch_idx, batch, metrics, part):
        """
        Run batch through the model, compute metrics, and
        save predictions to disk.

        Save directory is defined by save_path in the inference
        config and current partition.

        Args:
            batch_idx (int): the index of the current batch.
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type
                of the partition (train or inference).
            part (str): name of the partition. Used to define proper saving
                directory.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform)
                and model outputs.
        """

        def back_to_hw(x, orig_h, orig_w, h=380, w=507):
            if orig_h == h and orig_w == w:
                return x
            if orig_h <= h and orig_w <= w:
                return x[:, :orig_h, :orig_w]
            x = F.interpolate(x.unsqueeze(0), (orig_h, orig_w), mode="bilinear", align_corners=False).squeeze(0)
            return x

        def crop_(x, orig_h, orig_w):
            return x[:, :orig_h, :orig_w]
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        outputs = self.model(**batch)
        batch.update(outputs)
        lensed = batch["lensed"][0]
        lensed_exist = False
        if lensed is not None:
            lensed_exist = True

        if metrics is not None and lensed_exist:
            for met in self.metrics["inference"]:
                metrics.update(met.name, met(**batch))

        # Some saving logic. This is an example
        # Use if you need to save predictions on disk

        batch_size = batch["lensless"].shape[0]

        for i in range(batch_size):
            # clone because of
            # https://github.com/pytorch/pytorch/issues/1995
            cur_id = batch["id"][i]
            orig_hw = batch["orig_hw"][i].clone()
            orig_h, orig_w = orig_hw[0], orig_hw[1]
            reconstructed = batch["reconstructed"][i].clone()
            lensless = batch["lensless"][i].clone()
            psf = batch["psf"][i].clone()
            lensed = batch["lensed"][i]
            if lensed is not None:
                lensed = batch["lensed"][i].clone()

            output = {
                "reconstructed": reconstructed,
                "lensless": lensless,
                "psf": psf
            }
            if lensed_exist:
                output["lensed"] = lensed

            cur_path = self.save_path
            cur_path.mkdir(parents=True, exist_ok=True)

            if self.save_path is not None:
                # you can use safetensors or other lib here
                path_to_save = cur_path / cur_id
                path_to_save.mkdir(parents=True, exist_ok=True)

                torchvision.utils.save_image(back_to_hw(output["lensless"], orig_h, orig_w), path_to_save / f"lensless.png")
                if lensed_exist:
                    lensed_orig_hw = batch["lensed_orig_hw"][i].clone()
                    lensed_orig_h, lensed_orig_w = lensed_orig_hw[0], lensed_orig_hw[1]
                    torchvision.utils.save_image(get_roi_bchw(crop_(output["lensed"], lensed_orig_h, lensed_orig_w).unsqueeze(0)).squeeze(0),
                                                 path_to_save / f"lensed_roi.png")

                torchvision.utils.save_image(get_roi_bchw(back_to_hw(output["reconstructed"], orig_h, orig_w).unsqueeze(0)).squeeze(0),
                                             path_to_save / f"reconstructed_roi.png")

        return batch

    def _inference_part(self, part, dataloader):
        """
        Run inference on a given partition and save predictions

        Args:
            part (str): name of the partition.
            dataloader (DataLoader): dataloader for the given partition.
        Returns:
            logs (dict): metrics, calculated on the partition.
        """

        self.is_train = False
        self.model.eval()

        self.evaluation_metrics.reset()

        # create Save dir
        if self.save_path is not None:
            Path(self.save_path).mkdir(exist_ok=True, parents=True)

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                    part=part,
                    metrics=self.evaluation_metrics,
                )

        return self.evaluation_metrics.result()
