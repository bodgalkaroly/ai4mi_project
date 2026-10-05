#!/usr/bin/env python3

"""Remove small connected components from stitched predictions.

Groups a prediction directory's slices into one volume per patient, drops the
connected components smaller than --min_voxels for each class, and writes the
cleaned slices back out under the same filenames, so evaluate.py reads them
unchanged.

Slice-wise prediction leaves isolated specks far from the organ. Dice barely
notices them, being volume weighted, while Hausdorff is a maximum, so a single
stray voxel can set the score on its own.

Every component above the threshold is kept, not just the largest: these organs
are elongated and the prediction breaks along their length, so the smaller
pieces are usually more of the real organ rather than noise. The same caution
applies to the threshold itself -- set it too high and it deletes those pieces,
leaving stretches of the organ unpredicted and Hausdorff worse than before.

    $ python post_process.py --pred_dir results/segthor_clean/cedice/best_epoch/val \
        --dest results/segthor_clean/cedice/best_epoch/val_clean
"""

import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
from scipy.ndimage import label

K: int = 5


def read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        mask = np.asarray(image)
    if mask.ndim != 2 or np.any(mask > 252) or np.any(mask % 63):
        raise ValueError(f"error with {path}")
    return mask // 63


def drop_small(mask: np.ndarray, min_voxels: int) -> np.ndarray:
    """Every connected component of at least min_voxels, or the mask untouched
    if that would empty it — a class made only of specks is left for the metrics
    to judge rather than silently deleted."""
    labelled, n = label(mask)
    if n <= 1:
        return mask

    sizes = np.bincount(labelled.ravel())
    sizes[0] = 0  # Background is not a component
    keep = np.flatnonzero(sizes >= min_voxels)

    return np.isin(labelled, keep) if keep.size else mask


def main() -> None:
    parser = argparse.ArgumentParser(description="drop small connected components "
                                                 "from segthor predictions")
    parser.add_argument("--pred_dir", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--min_voxels", type=int, default=50,
                        help="Components smaller than this are removed (default: 50).")
    args = parser.parse_args()

    patients: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for path in sorted(args.pred_dir.glob("*.png")):
        patient, _, index = path.stem.rpartition("_")
        patients[patient].append((int(index), path))
    if not patients:
        raise ValueError(f"no predictions in {args.pred_dir}")

    args.dest.mkdir(parents=True, exist_ok=True)

    for patient, slices in sorted(patients.items()):
        slices.sort()
        if [index for index, _ in slices] != list(range(len(slices))):
            raise ValueError(f"error with {patient}")

        volume = np.stack([read_mask(path) for _, path in slices], axis=-1)

        cleaned = np.zeros_like(volume)
        removed: int = 0
        for k in range(1, K):
            kept = drop_small(volume == k, args.min_voxels)
            removed += int((volume == k).sum() - kept.sum())
            cleaned[kept] = k

        for z, (_, path) in enumerate(slices):
            Image.fromarray((cleaned[..., z] * 63).astype(np.uint8)).save(args.dest / path.name)

        print(f"{patient}: removed {removed} voxels")


if __name__ == "__main__":
    main()
