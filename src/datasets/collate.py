import torch
import torch.nn.functional as F

def pad_crop(x, H, W):
    c, h, w = x.shape
    if h == H and w == W:
        return x
    if h <= H and w <= W:
        pad_bottom = H - h
        pad_right = W - w
        x = F.pad(x, (0, pad_right, 0, pad_bottom), mode='constant', value=0)
    else:
        x = F.interpolate(x.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False).squeeze(0)
    return x

def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = {}
    H = 380
    W = 507
    if dataset_items[0]["lensed"] is None:
        result_batch["lensed"] = [None] * len(dataset_items)
    else:
        result_batch["lensed"] = torch.stack(
            [pad_crop(elem["lensed"], H, W) for elem in dataset_items], dim=0
        )
    result_batch["lensless"] = torch.stack(
        [pad_crop(elem["lensless"], H, W) for elem in dataset_items], dim=0
    )
    result_batch["psf"] = torch.stack(
        [pad_crop(elem["psf"], H, W) for elem in dataset_items], dim=0
    )
    if "orig_h" in dataset_items[0]:
        result_batch["orig_hw"] = torch.tensor(
            [[elem["orig_h"], elem["orig_w"]] for elem in dataset_items]
        )
    if "id" in dataset_items[0]:
        result_batch["id"] = [elem["id"] for elem in dataset_items]
    return result_batch
