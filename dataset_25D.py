#!/usr/bin/env python3

# MIT License

# Copyright (c) 2025 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from pathlib import Path
from typing import Callable, Union

from PIL import Image
from torch.utils.data import Dataset

from torch import Tensor
import torch

def make_dataset(root, subset) -> list[tuple[Path, Path | None]]:
    assert subset in ['train', 'val', 'test']

    root = Path(root)
    print(f"> {root=}")

    img_path = root / subset / 'img'
    full_path = root / subset / 'gt'

    images: list[Path] = sorted(img_path.glob("*.png"))
    full_labels: list[Path | None]
    if subset != 'test':
        full_labels = sorted(full_path.glob("*.png"))
    else:
        full_labels = [None] * len(images)

    return list(zip(images, full_labels))


class SliceDataset(Dataset):
    def __init__(
            self,
            subset,
            root_dir,
            img_transform=None,
            gt_transform=None,
            augment=None,
            geometric_augment=None,
            equalize=False,
            debug=False
    ):
        self.root_dir: str = root_dir
        self.img_transform: Callable = img_transform
        self.gt_transform: Callable = gt_transform
        self.intensity_augmentation: Callable = augment # added intensity augmentation
        self.geometric_augmentation: Callable = geometric_augment   # added geometric augmentation
        self.equalize: bool = equalize

        self.test_mode: bool = subset == 'test'

        self.files = make_dataset(root_dir, subset)

        if debug:
            self.files = self.files[:10]

        self.patient_slices = {}

        for img_path, gt_path in self.files:
            patient_id, slice_id = img_path.stem.rsplit("_", 1)
            slice_id = int(slice_id)

            if patient_id not in self.patient_slices:
                self.patient_slices[patient_id] = []

            self.patient_slices[patient_id].append(
                (slice_id, img_path, gt_path)
            )

        for patient_id in self.patient_slices:
            self.patient_slices[patient_id].sort(
                key=lambda x: x[0]
            )



        print(f">> Created {subset} dataset with {len(self)} images...")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index) -> dict[str, Union[Tensor, int, str]]:
        img_path, gt_path = self.files[index]

        # ------------------------------------------------------------
        # Identify patient and center slice
        # ------------------------------------------------------------

        patient_id, center_slice_id = img_path.stem.rsplit("_", 1)
        center_slice_id = int(center_slice_id)

        patient_slices = self.patient_slices[patient_id]

        # Find the position of the center slice in this patient's volume
        center_position = next(
            i
            for i, (slice_id, _, _) in enumerate(patient_slices)
            if slice_id == center_slice_id
        )

        # ------------------------------------------------------------
        # Load 5-slice context:
        # [z-2, z-1, z, z+1, z+2]
        #
        # At the volume boundaries, replicate the edge slice.
        # ------------------------------------------------------------

        neighbor_images = []

        for offset in range(-2, 3):
            neighbor_position = min(
                max(center_position + offset, 0),
                len(patient_slices) - 1
            )

            _, neighbor_img_path, _ = patient_slices[neighbor_position]

            neighbor_img = Image.open(
                neighbor_img_path
            ).convert("L")

            # Intensity augmentation, if used
            if self.intensity_augmentation is not None:
                neighbor_img = self.intensity_augmentation(
                    neighbor_img
                )

            neighbor_img = self.img_transform(neighbor_img)

            # img_transform returns [1, W, H]
            neighbor_images.append(neighbor_img)

        # Stack into [5, W, H]
        img = torch.cat(neighbor_images, dim=0)

        # ------------------------------------------------------------
        # Load GT of CENTER slice only
        # ------------------------------------------------------------

        data_dict = {
            "images": img,
            "stems": img_path.stem
        }

        if not self.test_mode:
            gt = Image.open(gt_path).convert("L")

            # Geometric augmentation is deliberately not applied here.
            # It requires a corresponding implementation for 5-channel
            # 2.5D images.

            gt = self.gt_transform(gt)

            _, W, H = img.shape
            K, _, _ = gt.shape

            assert img.shape == (5, W, H)
            assert gt.shape == (K, W, H)

            data_dict["gts"] = gt

        return data_dict