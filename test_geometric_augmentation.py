import torch
import numpy as np
from pathlib import Path
from PIL import Image

from augmentations import GeometricAugmentation
from dataset import SliceDataset
from augmentations_main import img_transform, gt_transform


# ============================================================
# 1. Direct augmentation test on Patient_02_0150
# ============================================================

image_path = Path(
    "data/SEGTHOR_FULL_SLICED/train/img/Patient_02_0150.png"
)
gt_path = Path(
    "data/SEGTHOR_FULL_SLICED/train/gt/Patient_02_0150.png"
)

image = Image.open(image_path).convert("L")
gt = Image.open(gt_path).convert("L")

geometric_augmentation = GeometricAugmentation(
    rotation_prob=1.0,
    rotation_range=(-15, 15),
    scaling_prob=1.0,
    scaling_range=(0.9, 1.1),
    translation_prob=1.0,
    translation_range=(-0.1, 0.1),
)

aug1_image, aug1_gt = geometric_augmentation(image, gt)
aug2_image, aug2_gt = geometric_augmentation(image, gt)

print("\n===== DIRECT AUGMENTATION TEST =====")

print(
    "Image difference:",
    torch.tensor(np.asarray(aug1_image), dtype=torch.float32)
    .sub(torch.tensor(np.asarray(aug2_image), dtype=torch.float32))
    .abs()
    .mean()
    .item()
)

print(
    "GT difference:",
    torch.tensor(np.asarray(aug1_gt), dtype=torch.float32)
    .sub(torch.tensor(np.asarray(aug2_gt), dtype=torch.float32))
    .abs()
    .mean()
    .item()
)

print("Original GT unique values:", np.unique(np.asarray(gt)))
print("Augmented GT unique values:", np.unique(np.asarray(aug1_gt)))


# ============================================================
# 2. Test through SliceDataset
# ============================================================

root_dir = Path("data") / "SEGTHOR_FULL_SLICED"

dataset = SliceDataset(
    "train",
    root_dir,
    img_transform=img_transform,
    gt_transform=lambda img: gt_transform(5, img),
    geometric_augment=geometric_augmentation,
    debug=False,
)


# Find Patient_02_0150 in the dataset
target_stem = "Patient_02_0150"

target_index = None

for i, (img_path, gt_path) in enumerate(dataset.files):
    if img_path.stem == target_stem:
        target_index = i
        break

assert target_index is not None, f"{target_stem} not found"

print("\n===== DATASET TEST =====")
print("Target index:", target_index)
print("Target stem:", dataset.files[target_index][0].stem)


sample_1 = dataset[target_index]
sample_2 = dataset[target_index]


print("\nSample:")
print("Image shape:", sample_1["images"].shape)
print("Image dtype:", sample_1["images"].dtype)
print("Image min:", sample_1["images"].min().item())
print("Image max:", sample_1["images"].max().item())

print("GT shape:", sample_1["gts"].shape)
print("GT dtype:", sample_1["gts"].dtype)
print("GT unique values:", torch.unique(sample_1["gts"]))
print("Stem:", sample_1["stems"])


# Compare two independently augmented accesses
image_difference = torch.abs(
    sample_1["images"] - sample_2["images"]
).mean()

gt_difference = torch.abs(
    sample_1["gts"].float() - sample_2["gts"].float()
).mean()


print("\n===== DATASET RANDOMNESS TEST =====")
print("Mean image difference:", image_difference.item())
print("Mean GT difference:", gt_difference.item())