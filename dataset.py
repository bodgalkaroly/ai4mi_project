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

from torch import Tensor
from PIL import Image
from torch.utils.data import Dataset


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

        print(f">> Created {subset} dataset with {len(self)} images...")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index) -> dict[str, Union[Tensor, int, str]]:
        img_path, gt_path = self.files[index]

        # Load image
        img = Image.open(img_path).convert("L")

        # Load GT if available
        if not self.test_mode:
            gt = Image.open(gt_path).convert("L")

            # Geometric augmentation must be applied
            # to image and GT using the same transformation.
            if self.geometric_augmentation is not None:
                img, gt = self.geometric_augmentation(img, gt)

        # Intensity augmentation is applied only to the image.
        if self.intensity_augmentation is not None:
            img = self.intensity_augmentation(img)

        # Convert to tensors
        img: Tensor = self.img_transform(img)

        data_dict = {
            "images": img,
            "stems": img_path.stem
        }

        if not self.test_mode:
            gt: Tensor = self.gt_transform(gt)

            _, W, H = img.shape
            K, _, _ = gt.shape

            assert gt.shape == (K, W, H)

            data_dict["gts"] = gt

        return data_dict
