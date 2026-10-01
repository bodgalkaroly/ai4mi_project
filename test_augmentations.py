# # test_augmentations.py
#
# from pathlib import Path
#
# from PIL import Image
#
# from augmentations import IntensityAugmentation
#
# import numpy as np
#
# # ---------------------------------------------------------
# # Paths
# # ---------------------------------------------------------
#
# image_dir = Path("data/SEGTHOR_FULL_SLICED/train/img")
# output_dir = Path("augmentation_test")
#
# output_dir.mkdir(parents=True, exist_ok=True)
#
#
# # ---------------------------------------------------------
# # Select one real CT slice
# # ---------------------------------------------------------
#
# images = sorted(image_dir.glob("*.png"))
#
# if not images:
#     raise RuntimeError(f"No PNG images found in {image_dir}")
#
# image_path = images[0]
#
# print(f"Using image: {image_path}")
#
#
# # ---------------------------------------------------------
# # Load original image
# # ---------------------------------------------------------
#
# image = Image.open(image_path).convert("L")
#
# image.save(output_dir / "original.png")
#
#
# # ---------------------------------------------------------
# # 1. Gaussian noise only
# # ---------------------------------------------------------
#
# noise_aug = IntensityAugmentation(
#     noise_prob=1.0,
#     noise_std=0.03,
#
#     lowres_prob=0.0,
#
#     intensity_prob=0.0,
# )
#
# noise_image = noise_aug(image)
# noise_image.save(output_dir / "gaussian_noise.png")
#
#
# # ---------------------------------------------------------
# # 2. Simulated low resolution only
# # ---------------------------------------------------------
#
# lowres_aug = IntensityAugmentation(
#     noise_prob=0.0,
#
#     lowres_prob=1.0,
#     lowres_scale_range=(0.5, 0.75),
#
#     intensity_prob=0.0,
# )
#
# lowres_image = lowres_aug(image)
# lowres_image.save(output_dir / "low_resolution.png")
#
#
# # ---------------------------------------------------------
# # 3. Intensity scaling only
# # ---------------------------------------------------------
#
# intensity_aug = IntensityAugmentation(
#     noise_prob=0.0,
#
#     lowres_prob=0.0,
#
#     intensity_prob=1.0,
#     intensity_scale_range=(0.8, 1.2),
# )
#
# intensity_image = intensity_aug(image)
# intensity_image.save(output_dir / "intensity_scaling.png")
#
#
# # ---------------------------------------------------------
# # 4. All augmentations
# # ---------------------------------------------------------
#
# combined_aug = IntensityAugmentation(
#     noise_prob=1.0,
#     noise_std=0.03,
#
#     lowres_prob=1.0,
#     lowres_scale_range=(0.5, 0.75),
#
#     intensity_prob=1.0,
#     intensity_scale_range=(0.8, 1.2),
# )
#
# combined_image = combined_aug(image)
# combined_image.save(output_dir / "combined.png")
#
#
# #-------5. Stats
# def print_stats(name, image):
#     image_array = np.asarray(image).astype(np.float32) / 255.0
#
#     print(f"\n{name}")
#     print(f"  shape:      {image_array.shape}")
#     print(f"  min:        {image_array.min():.4f}")
#     print(f"  max:        {image_array.max():.4f}")
#     print(f"  mean:       {image_array.mean():.4f}")
#     print(f"  std:        {image_array.std():.4f}")
#     print(f"  zeros:      {(image_array == 0).mean() * 100:.2f}%")
#     print(f"  ones:       {(image_array == 1).mean() * 100:.2f}%")
# #-------
#
#
# print(f"\nSaved augmentation examples to: {output_dir}")
#
#
#
# print_stats("Original", image)
# print_stats("Gaussian noise", noise_image)
# print_stats("Low resolution", lowres_image)
# print_stats("Intensity scaling", intensity_image)
# print_stats("Combined", combined_image)


from functools import partial
from pathlib import Path
import torch
from main import img_transform, gt_transform
from dataset import SliceDataset
from augmentations import IntensityAugmentation


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

root_dir = Path("data") / "SEGTHOR_FULL_SLICED"


# ---------------------------------------------------------
# Intensity augmentation
# ---------------------------------------------------------

augmentation = IntensityAugmentation(
    noise_prob=1.0,
    noise_std=0.03,

    lowres_prob=0.0,

    intensity_prob=0.0,
)


# ---------------------------------------------------------
# Create training dataset
# ---------------------------------------------------------

dataset = SliceDataset(
    subset="train",
    root_dir=root_dir,
    img_transform=img_transform,
    gt_transform=partial(gt_transform, 5),
    augment=augmentation,
)


# ---------------------------------------------------------
# Load one sample
# ---------------------------------------------------------

sample = dataset[0]


# ---------------------------------------------------------
# Check results
# ---------------------------------------------------------

print("\nSample:")
print("Image shape:", sample["images"].shape)
print("Image dtype:", sample["images"].dtype)
print("Image min:", sample["images"].min().item())
print("Image max:", sample["images"].max().item())

print("GT shape:", sample["gts"].shape)
print("GT dtype:", sample["gts"].dtype)

print("Stem:", sample["stems"])

print("\nTesting online randomness...")

sample_1 = dataset[0]
sample_2 = dataset[0]

difference = torch.abs(
    sample_1["images"] - sample_2["images"]
).mean()

print("Mean absolute difference:", difference.item())