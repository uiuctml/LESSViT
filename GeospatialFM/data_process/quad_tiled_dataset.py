import torch

from .transforms import FixedQuadCropAll


class QuadTiledDataset(torch.utils.data.Dataset):
    """Wraps a base enmap dataset (constructed with crop_size=None, i.e. its own
    transform normalizes/augments but does not crop, so __getitem__ returns the
    full native 2*crop_size x 2*crop_size tensors) so that each native image
    yields its 4 fixed non-overlapping crop_size x crop_size quadrants as 4
    separate training samples -- every corner systematically covered every epoch,
    instead of one RandomCropAll draw per image per epoch (covered only in
    expectation over many epochs).

    Each of the 4 quadrants still gets its own independent flip/rotation draw:
    the base dataset's __getitem__ is called once per quadrant (a fresh random
    augmentation draw each call, since is_train's flip/rotation logic runs before
    this wrapper ever sees the tensor), only the crop *position* is made
    deterministic instead of random.

    `has_spatial_label` must be True for segmentation (label is a per-pixel mask,
    cropped alongside optical/radar) and False for classification/multilabel
    (label is a per-image scalar/one-hot vector with no spatial extent -- cropping
    it would error, and it shouldn't change per quadrant anyway).
    """

    def __init__(self, base_dataset, crop_size, has_spatial_label):
        self.base_dataset = base_dataset
        self.crop_size = crop_size
        self.has_spatial_label = has_spatial_label

    def __len__(self):
        return 4 * len(self.base_dataset)

    def __getitem__(self, index):
        base_idx, quadrant_idx = divmod(index, 4)
        example = dict(self.base_dataset[base_idx])
        label = example.get("label") if self.has_spatial_label else None
        optical, radar, label = FixedQuadCropAll(
            example.get("optical"), example.get("radar"), label,
            crop_size=self.crop_size, quadrant_idx=quadrant_idx,
        )
        if optical is not None:
            example["optical"] = optical
        if radar is not None:
            example["radar"] = radar
        if label is not None:
            example["label"] = label
        return example
