from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json
from PIL import Image

class CustomDirDataset(BaseDataset):
    """
    Example of a nested dataset class to show basic structure.

    Uses random vectors as objects and random integers between
    0 and n_classes-1 as labels.
    """

    def __init__(self, path, *args, **kwargs):
        index_path = ROOT_PATH / path / "index.json"

        # each nested dataset class must have an index field that
        # contains list of dicts. Each dict contains information about
        # the object, including label, path, etc.
        if index_path.exists():
            index = read_json(str(index_path))
        else:
            index = self._create_index(path)

        super().__init__(index, mode="inference", *args, **kwargs)

    def _create_index(self, path):
        """
        Create index for the dataset. The function processes dataset metadata
        and utilizes it to get information dict for each element of
        the dataset.

        Args:
            name (str): partition name
        Returns:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        """
        def path_check(path, suff=".png"):
            if (not str(path.stem).startswith("ImageID")) or path.suffix != suff:
                return False
            if not str(path.stem)[7:].isnumeric():
                return False
            return True

        index = []
        data_path = ROOT_PATH / path
        if not data_path.exists():
            raise FileNotFoundError(f"no data: {data_path}")
        data_path.mkdir(exist_ok=True, parents=True)

        lensless_path = data_path / "lensless"
        masks_path = data_path / "masks"
        lensed_path = data_path / "lensed"

        lensed_exists = lensed_path.exists()

        all_lensless_files = list(lensless_path.iterdir())
        all_mask_files = list(masks_path.iterdir())
        if lensed_exists:
            all_lensed_files = list(lensed_path.iterdir())
        else:
            all_lensed_files = [None] * len(all_lensless_files)

        ids = dict()
        def put_path_in_id(path, name):
            cur_id = int(str(path.stem)[7:])
            if cur_id not in ids:
                ids[cur_id] = dict()
                ids[cur_id][name] = path
            else:
                if name in ids[cur_id]:
                    return False
                else:
                    ids[cur_id][name] = path
            return True

        def put_from_list(all_files, name, suff=".png"):
            for f in all_files:
                assert path_check(f, suff)
                put_path_in_id(f, name)

        put_from_list(all_lensless_files, "lensless", suff=".png")
        put_from_list(all_mask_files, "mask", suff=".npy")
        if lensed_exists:
            put_from_list(all_lensed_files, "lensed", suff=".png")
        for cur_id in ids.keys():
            assert "lensless" in ids[cur_id]
            assert "mask" in ids[cur_id]
            if "lensed" in ids[cur_id]:
                index.append({
                    "id": f"ImageID{cur_id}",
                    "lensless": str(ids[cur_id]["lensless"]),
                    "mask": str(ids[cur_id]["mask"]),
                    "lensed": str(ids[cur_id]["lensed"]),
                })
            else:
                index.append({
                    "id": f"ImageID{cur_id}",
                    "lensless": str(ids[cur_id]["lensless"]),
                    "mask": str(ids[cur_id]["mask"]),
                    "lensed": None,
                })
        write_json(index, str(data_path / "index.json"))
        return index

