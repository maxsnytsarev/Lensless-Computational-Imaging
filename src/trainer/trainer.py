from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from lensless_helpers.preprocessor import get_roi_bchw


def get_log(batch):
    lensless = batch["lensless"][0].detach().cpu().numpy().transpose(1, 2, 0)
    lensed = batch["lensed"][0].detach().cpu().numpy().transpose(1, 2, 0)
    reconstructed = batch["reconstructed"][0].detach().cpu().numpy().transpose(1, 2, 0)
    psf = batch["psf"][0].detach().cpu()
    psf = psf.clone() / (psf.abs().max()+1e-8)
    psf = psf.numpy().transpose(1, 2, 0)
    lensed_roi = get_roi_bchw(batch["lensed"])[0].detach().cpu().numpy().transpose(1, 2, 0)
    reconstructed_roi = get_roi_bchw(batch["reconstructed"])[0].detach().cpu().numpy().transpose(1, 2, 0)
    return lensless, lensed, reconstructed, psf, lensed_roi, reconstructed_roi


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        outputs = self.model(**batch)
        batch.update(outputs)

        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if self.is_train:
            batch["loss"].backward()  # sum of all losses is always called loss
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            lensless, lensed, reconstructed, psf, lensed_roi, reconstructed_roi = get_log(batch)
            if self.writer is not None:
                self.writer.add_image(f"{mode}/lensless", lensless)
                self.writer.add_image(f"{mode}/lensed", lensed)
                self.writer.add_image(f"{mode}/reconstructed", reconstructed)
                self.writer.add_image(f"{mode}/PSF", psf)
                self.writer.add_image(f"{mode}/lensed_roi", lensed_roi)
                self.writer.add_image(f"{mode}/reconstructed_roi", reconstructed_roi)
        else:
            lensless, lensed, reconstructed, psf, lensed_roi, reconstructed_roi = get_log(batch)
            if self.writer is not None:
                self.writer.add_image(f"{mode}/lensless", lensless)
                self.writer.add_image(f"{mode}/lensed", lensed)
                self.writer.add_image(f"{mode}/reconstructed", reconstructed)
                self.writer.add_image(f"{mode}/PSF", psf)
                self.writer.add_image(f"{mode}/lensed_roi", lensed_roi)
                self.writer.add_image(f"{mode}/reconstructed_roi", reconstructed_roi)
