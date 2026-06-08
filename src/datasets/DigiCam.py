import numpy as np
import torch
from tqdm.auto import tqdm

import datasets
from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class DigiCamDataset(BaseDataset):
    def __init__(
        self,
        dataset_name="bezzam/DigiCam-Mirflickr-MultiMask-10K",
        columns=("lensless", "lensed", "mask_label"),
        split="train",
        *args,
        **kwargs
    ):
        self.labels = columns
        self.cur_dataset = datasets.load_dataset(dataset_name)[split]
        index = self.cur_dataset
        super().__init__(index, repo=dataset_name, columns=columns, *args, **kwargs)
