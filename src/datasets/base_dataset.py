import logging
import random
from typing import List

import numpy as np
import safetensors
import safetensors.torch
import torch
from huggingface_hub import hf_hub_download
from torch.utils.data import Dataset
from PIL import Image

from lensless_helpers.preprocessor import get_dataset_object, get_image_only, no_lensed_get

logger = logging.getLogger(__name__)


def to_shape(x):
    if len(x.shape) == 4:
        x = x.squeeze(0)
    x = x.permute(2, 0, 1)
    return x


class BaseDataset(Dataset):
    """
    Base class for the datasets.

    Given a proper index (HuggingFace Dataset), allows to process different datasets
    for the same task in the identical manner. Therefore, to work with
    several datasets, the user only have to define index in a nested class.
    """

    def __init__(
        self,
        index,
        repo="bezzam/DigiCam-Mirflickr-MultiMask-10K",
        limit=None,
        shuffle_index=False,
        instance_transforms=None,
        columns=("img", "label"),
        mode="train",
    ):
        self.columns = columns
        self.repo = repo
        self.cols = len(columns)
        self._assert_index_is_valid(index, mode)
        index = self._shuffle_and_limit_index(index, limit, shuffle_index)
        self.mask_cashe = dict()
        self._index: List[dict] = index
        self.mode = mode

        self.instance_transforms = instance_transforms

    def __getitem__(self, ind):
        """
        Get element from the index, preprocess it, and combine it
        into a dict.

        Notice that the choice of key names is defined by the template user.
        However, they should be consistent across dataset getitem, collate_fn,
        loss_function forward method, and model forward method.

        Args:
            ind (int): index in the self.index list.
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        """
        instance_data = dict()
        data_dict = self._index[ind]
        if self.mode == "train":
            for i in range(self.cols):
                instance_data[self.columns[i]] = data_dict[self.columns[i]]
            assert "mask_label" in instance_data.keys()
            m_label = instance_data["mask_label"]
            if not m_label in self.mask_cashe.keys():
                path = hf_hub_download(
                    repo_id=self.repo,
                    repo_type="dataset",
                    filename=f"masks/mask_{m_label}.npy",
                )
                mask = np.load(path)
                my_lensed, my_lensless, my_psf = get_dataset_object(
                    instance_data["lensed"],
                    instance_data["lensless"],
                    mask,
                )
                my_lensed = to_shape(my_lensed)
                my_lensless = to_shape(my_lensless)
                my_psf = to_shape(my_psf)
                instance_data["lensed"] = my_lensed
                instance_data["lensless"] = my_lensless
                instance_data["psf"] = my_psf
                self.mask_cashe[m_label] = (mask, my_psf)
            else:
                mask = self.mask_cashe[m_label][0]
                my_lensed, my_lensless = get_image_only(
                    instance_data["lensed"], instance_data["lensless"]
                )
                instance_data["lensed"] = to_shape(my_lensed)
                instance_data["lensless"] = to_shape(my_lensless)
                instance_data["psf"] = self.mask_cashe[m_label][1]
            instance_data["id"] = f"ID{ind}"
            instance_data = self.preprocess_data(instance_data)
        elif self.mode == "inference":
            cur_id = data_dict["id"]
            lensless_path = data_dict["lensless"]
            masks_path = data_dict["mask"]
            lensed_path = data_dict["lensed"]
            lensed_exists = True if lensed_path is not None else False
            lensless = Image.open(lensless_path)
            mask = np.load(masks_path)
            lensed = None
            if lensed_exists:
                lensed = Image.open(lensed_path)
            if lensed_exists:
                my_lensed, my_lensless, my_psf = get_dataset_object(
                    lensed,
                    lensless,
                    mask
                )
            else:
                my_lensed = None
                my_lensless, my_psf = no_lensed_get(
                    lensless,
                    mask
                )
            if my_lensed is not None:
                my_lensed = to_shape(my_lensed)
            my_lensless = to_shape(my_lensless)
            my_psf = to_shape(my_psf)
            instance_data["id"] = cur_id
            instance_data["lensed"] = my_lensed
            instance_data["lensless"] = my_lensless
            instance_data["orig_h"] = my_lensless.shape[1]
            instance_data["orig_w"] = my_lensless.shape[2]
            instance_data["psf"] = my_psf
        else:
            raise NotImplementedError
        return instance_data

    def __len__(self):
        """
        Get length of the dataset (length of the index).
        """
        return len(self._index)

    def preprocess_data(self, instance_data):
        """
        Preprocess data with instance transforms.

        Each tensor in a dict undergoes its own transform defined by the key.

        Args:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element) (possibly transformed via
                instance transform).
        """
        if self.instance_transforms is not None:
            for transform_name in self.instance_transforms.keys():
                instance_data[transform_name] = self.instance_transforms[
                    transform_name
                ](instance_data[transform_name])
        return instance_data

    @staticmethod
    def _filter_records_from_dataset(
        index: list,
    ) -> list:
        """
        Filter some of the elements from the dataset depending on
        some condition.

        This is not used in the example. The method should be called in
        the __init__ before shuffling and limiting.

        Args:
            index (Dataset): HuggingFace Dataset, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and image.
        Returns:
            index (Dataset): HuggingFace Dataset, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and image.
        """
        # Filter logic
        pass

    @staticmethod
    def _assert_index_is_valid(index, mode):
        """
        Check the structure of the index and ensure it satisfies the desired
        conditions.

        Args:
            index (Dataset): HuggingFace Dataset, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and image.
        """
        for entry in index:
            assert "lensless" in entry, (
                "Each dataset item should include field 'lensless'" " - noisy image."
            )
            assert "lensed" in entry, (
                "Each dataset item should include field 'lensless'"
                " - object ground-truth image."
            )
            if mode == "train":
                assert (
                    "mask_label" in entry
                ), "Each dataset item should include field 'mask_label'"
            else:
                assert (
                    "mask" in entry
                ), "Each dataset item should include field 'mask'"

    @staticmethod
    def _sort_index(index):
        """
        Sort index via some rules.

        This is not used in the example. The method should be called in
        the __init__ before shuffling and limiting and after filtering.

        Args:
            index (Dataset): HuggingFace Dataset, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and image.
        Returns:
            index (Dataset): HuggingFace Dataset, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and image.
        """
        return index.sort("KEY_FOR_SORTING")

    @staticmethod
    def _shuffle_and_limit_index(index, limit, shuffle_index):
        """
        Shuffle elements in index and limit the total number of elements.

        Args:
            index (Dataset): HuggingFace Dataset, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and image.
            limit (int | None): if not None, limit the total number of elements
                in the dataset to 'limit' elements.
            shuffle_index (bool): if True, shuffle the index. Uses python
                random package with seed 42.
        """
        if shuffle_index:
            index = index.shuffle(seed=42)

        if limit is not None:
            index = index.select(range(limit))
        return index
