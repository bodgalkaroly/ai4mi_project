#!/usr/bin/env python3.7

# MIT License

# Copyright (c) 2024 Hoel Kervadec

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

import pickle
import random
import argparse
import warnings
from pathlib import Path
from functools import partial
from multiprocessing import Pool

import numpy as np
import nibabel as nib
from scipy import ndimage
from skimage.io import imsave
from skimage.transform import resize

from utils import map_, tqdm_


# ============================================================================
# Sanity checks
# ============================================================================

def sanity_ct(ct, x, y, z, dx, dy, dz) -> bool:
    assert ct.dtype in [np.int16, np.int32, np.float32, np.float64], ct.dtype
    assert -1000 <= ct.min(), ct.min()
    assert ct.max() <= 31743, ct.max()

    assert 0.896 <= dx <= 1.37, dx
    assert dx == dy
    assert 2 <= dz <= 3.7, dz

    assert (x, y) == (512, 512)
    assert x == y
    assert 135 <= z <= 284, z

    return True


def sanity_gt(gt, ct) -> bool:
    assert gt.shape == ct.shape
    assert gt.dtype in [np.uint8], gt.dtype

    return True


# ============================================================================
# Stage 1: intensity-domain handling
# ============================================================================

def make_body_mask(
    ct: np.ndarray,
    threshold: float = -500.0,
) -> np.ndarray:
    """
    Build a simple axial body-envelope mask.

    For each axial slice:
        1. threshold CT values
        2. keep largest connected component
        3. fill enclosed holes

    This is the mask method used for our scratch analysis.
    """
    assert ct.ndim == 3

    body = np.zeros(ct.shape, dtype=bool)

    for idz in range(ct.shape[2]):
        slice_mask = ct[:, :, idz] > threshold

        labels, num = ndimage.label(slice_mask)

        if num == 0:
            continue

        counts = np.bincount(labels.ravel())
        counts[0] = 0

        largest = int(np.argmax(counts))
        slice_mask = labels == largest

        slice_mask = ndimage.binary_fill_holes(slice_mask)

        body[:, :, idz] = slice_mask

    return body


def intensity_domain(
    ct: np.ndarray,
    use_body_mask: bool = False,
    body_threshold: float = -500.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Stage 1: intensity-domain preprocessing.

    Currently:
        - CT values remain in their original quantitative domain.
        - An optional body mask can be generated.

    No intensity normalization is performed here.
    No spatial normalization is performed here.
    No CT voxels are removed or modified using the body mask.
    """
    processed = ct.astype(np.float32, copy=True)

    body_mask = None

    if use_body_mask:
        body_mask = make_body_mask(
            processed,
            threshold=body_threshold,
        )

    return processed, body_mask


# ============================================================================
# Stage 2: intensity normalization
# ============================================================================

def fit_intensity_statistics(
    patient_ids: list[str],
    source_path: Path,
    body_threshold: float = -500.0,
    samples_per_patient: int = 20000,
    robust_low: float = 0.5,
    robust_high: float = 99.5,
    seed: int = 0,
) -> dict[str, float]:
    """
    Fit robust training-dataset-level intensity statistics.

    Statistics are calculated from body-mask voxels of the training
    patients only.

    The fitted parameters are:
        - lower clipping percentile
        - upper clipping percentile
        - mean after clipping
        - standard deviation after clipping

    A fixed maximum number of voxels is sampled per patient so that
    the complete training dataset does not need to be kept in memory.
    """

    rng = np.random.default_rng(seed)
    samples: list[np.ndarray] = []

    print(
        f"Fitting intensity statistics from "
        f"{len(patient_ids)} training patients"
    )

    for id_ in tqdm_(patient_ids):

        id_path = source_path / "train" / id_
        ct_path = id_path / f"{id_}.nii.gz"

        nib_obj = nib.load(str(ct_path))
        ct = np.asarray(nib_obj.dataobj).astype(np.float32)

        body_mask = make_body_mask(
            ct,
            threshold=body_threshold,
        )

        values = ct[body_mask]

        assert values.size > 0, (
            f"Empty body mask for {id_}"
        )

        if values.size > samples_per_patient:
            indices = rng.choice(
                values.size,
                size=samples_per_patient,
                replace=False,
            )
            values = values[indices]

        samples.append(
            values.astype(np.float32, copy=False)
        )

    values = np.concatenate(samples)

    clip_low, clip_high = np.percentile(
        values,
        [robust_low, robust_high],
    )

    clipped_values = np.clip(
        values,
        clip_low,
        clip_high,
    )

    mean = float(clipped_values.mean())
    std = float(clipped_values.std())

    assert std > 0

    statistics = {
        "clip_low": float(clip_low),
        "clip_high": float(clip_high),
        "mean": mean,
        "std": std,
        "robust_low": float(robust_low),
        "robust_high": float(robust_high),
        "samples_per_patient": int(samples_per_patient),
        "num_patients": len(patient_ids),
    }

    print()
    print("Training-dataset-level intensity statistics:")
    print(f"  clip low:   {clip_low:.4f}")
    print(f"  clip high:  {clip_high:.4f}")
    print(f"  mean:       {mean:.4f}")
    print(f"  std:        {std:.4f}")

    return statistics


def intensity_normalization(
    ct: np.ndarray,
    intensity_stats: dict[str, float],
) -> np.ndarray:
    """
    Stage 2: robust training-dataset-level z-score normalization.

    The CT is:
        1. clipped to the training-set body intensity range
        2. standardized using the training-set mean and standard deviation

    The same fitted parameters are applied to every patient.

    The body mask is used only when fitting the statistics. The
    normalization itself is applied to the complete CT volume.
    """

    clip_low = intensity_stats["clip_low"]
    clip_high = intensity_stats["clip_high"]
    mean = intensity_stats["mean"]
    std = intensity_stats["std"]

    clipped = np.clip(
        ct.astype(np.float32),
        clip_low,
        clip_high,
    )

    normalized = (
        (clipped - mean) / std
    )

    return normalized.astype(np.float32, copy=False)


# ============================================================================
# Stage 3: spatial normalization
# ============================================================================

def spatial_normalization(
    ct: np.ndarray,
    gt: np.ndarray,
    shape: tuple[int, int],
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Stage 3: reproduce the original SegTHOR spatial preprocessing.

    Each axial slice is resized from the original in-plane resolution
    to the requested output shape, normally 256 x 256.

    The z dimension is unchanged.

    CT:
        linear interpolation, preserve range, no anti-aliasing

    GT:
        nearest-neighbor interpolation
    """

    x, y, z = ct.shape

    spatial_ct = np.empty(
        (shape[0], shape[1], z),
        dtype=np.float32,
    )

    spatial_gt = np.empty(
        (shape[0], shape[1], z),
        dtype=np.uint8,
    )

    for idz in range(z):

        spatial_ct[:, :, idz] = resize(
            ct[:, :, idz],
            shape,
            mode="constant",
            preserve_range=True,
            anti_aliasing=False,
            order=1,
        ).astype(np.float32)

        spatial_gt[:, :, idz] = resize(
            gt[:, :, idz],
            shape,
            mode="constant",
            preserve_range=True,
            anti_aliasing=False,
            order=0,
        ).astype(np.uint8)

    return spatial_ct, spatial_gt


# ============================================================================
# Stage 4: tensor/storage normalization
# ============================================================================

def tensor_normalization(
    image: np.ndarray,
    intensity_stats: dict[str, float],
) -> np.ndarray:
    """
    Stage 4: convert the Stage 2 z-score image to a fixed [0, 1] range.

    The mapping is global and fixed for the entire dataset:
        Stage 2 clip_low  -> 0
        Stage 2 clip_high -> 1

    This is done before storing the slices as uint8 PNGs so that
    the existing SliceDataset can keep its /255 loading convention.
    """

    clip_low = intensity_stats["clip_low"]
    clip_high = intensity_stats["clip_high"]
    mean = intensity_stats["mean"]
    std = intensity_stats["std"]

    z_low = (clip_low - mean) / std
    z_high = (clip_high - mean) / std

    assert z_high > z_low

    normalized = np.clip(
        image,
        z_low,
        z_high,
    )

    normalized = (
        normalized - z_low
    ) / (z_high - z_low)

    normalized = normalized.astype(
        np.float32,
        copy=False,
    )

    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0

    return normalized


# ============================================================================
# Patient processing
# ============================================================================


def slice_patient(
    id_: str,
    dest_path: Path,
    source_path: Path,
    shape: tuple[int, int],
    use_body_mask: bool,
    body_threshold: float,
    intensity_stats: dict[str, float],
    test_mode: bool = False,
):

    id_path: Path = (
        source_path / ("train" if not test_mode else "test") / id_
    )

    ct_path: Path = (
        id_path / f"{id_}.nii.gz"
        if not test_mode
        else source_path / "test" / f"{id_}.nii.gz"
    )

    nib_obj = nib.load(str(ct_path))

    ct = np.asarray(nib_obj.dataobj)
    x, y, z = ct.shape
    dx, dy, dz = nib_obj.header.get_zooms()

    print(
        f"{id_}: "
        f"shape={ct.shape}, "
        f"spacing={(dx, dy, dz)}"
    )

    assert sanity_ct(
        ct,
        *ct.shape,
        *nib_obj.header.get_zooms(),
    )

    # ------------------------------------------------------------
    # Ground truth
    # ------------------------------------------------------------

    gt: np.ndarray

    if not test_mode:
        gt_path = id_path / "GT.nii.gz"
        gt_nib = nib.load(str(gt_path))
        gt = np.asarray(gt_nib.dataobj)

        assert sanity_gt(gt, ct)

    else:
        gt = np.zeros_like(ct, dtype=np.uint8)

    # ------------------------------------------------------------
    # Stage 1
    # ------------------------------------------------------------

    to_domain, body_mask = intensity_domain(
        ct,
        use_body_mask=use_body_mask,
        body_threshold=body_threshold,
    )

    # ------------------------------------------------------------
    # Stage 2
    # ------------------------------------------------------------
    to_intensity = intensity_normalization(
        to_domain,
        intensity_stats=intensity_stats,
    )

    # ------------------------------------------------------------
    # Stage 3
    # ------------------------------------------------------------


    to_spatial, to_spatial_gt = spatial_normalization(
        to_intensity,
        gt,
        shape,
    )

    # ------------------------------------------------------------
    # Stage 4
    # ------------------------------------------------------------

    to_tensor = tensor_normalization(
        to_spatial,
        intensity_stats=intensity_stats,
    )


    # ------------------------------------------------------------
    # Save slices
    # ------------------------------------------------------------

    for idz in range(to_tensor.shape[2]):

        img_slice = np.round(
            255.0 * to_tensor[:, :, idz]
        ).astype(np.uint8)

        gt_slice = to_spatial_gt[:, :, idz].astype(np.uint8)

        assert img_slice.shape == gt_slice.shape

        gt_slice *= 63

        assert gt_slice.dtype == np.uint8
        assert set(np.unique(gt_slice)) <= {
            0,
            63,
            126,
            189,
            252,
        }

        arrays = [img_slice, gt_slice]
        subfolders = ["img", "gt"]

        assert len(arrays) == len(subfolders)

        for save_subfolder, data in zip(
                subfolders,
                arrays,
        ):
            filename = f"{id_}_{idz:04d}.png"

            save_path = Path(
                dest_path,
                save_subfolder,
            )

            save_path.mkdir(
                parents=True,
                exist_ok=True,
            )

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                )

                imsave(
                    str(save_path / filename),
                    data,
                )

    return dx, dy, dz
# ============================================================================
# Splits
# ============================================================================

def get_splits(
    src_path: Path,
    retains: int,
    fold: int,
) -> tuple[list[str], list[str], list[str]]:

    ids = sorted(
        map_(
            lambda p: p.name,
            (src_path / "train").glob("*"),
        )
    )

    print(f"Founds {len(ids)} in the id list")
    print(ids[:10])

    assert len(ids) > retains

    random.shuffle(ids)

    validation_slice = slice(
        fold * retains,
        (fold + 1) * retains,
    )

    validation_ids = ids[validation_slice]

    assert len(validation_ids) == retains

    training_ids = [
        e for e in ids
        if e not in validation_ids
    ]

    assert (
        len(training_ids)
        + len(validation_ids)
        == len(ids)
    )

    test_ids = sorted(
        map_(
            lambda p: Path(p.stem).stem,
            (src_path / "test").glob("*"),
        )
    )

    print(f"Founds {len(test_ids)} test ids")
    print(test_ids[:10])

    return training_ids, validation_ids, test_ids


# ============================================================================
# Main
# ============================================================================

def main(args: argparse.Namespace):

    src_path = Path(args.source_dir)
    dest_path = Path(args.dest_dir)

    assert src_path.exists()
    assert not dest_path.exists()

    training_ids, validation_ids, test_ids = get_splits(
        src_path,
        args.retains,
        args.fold,
    )

    # ------------------------------------------------------------
    # Fit Stage 2 intensity-normalization parameters
    # using training patients only.
    # ------------------------------------------------------------

    assert args.body_mask, (
        "Stage 2 requires --body_mask to fit intensity statistics"
    )

    intensity_stats = fit_intensity_statistics(
        patient_ids=training_ids,
        source_path=src_path,
        body_threshold=args.body_threshold,
        samples_per_patient=args.samples_per_patient,
        robust_low=args.robust_low,
        robust_high=args.robust_high,
        seed=args.seed,
    )

    dest_path.mkdir(parents=True, exist_ok=True)

    with open(
        dest_path / "intensity_stats.pkl",
        "wb",
    ) as f:

        pickle.dump(
            intensity_stats,
            f,
            pickle.HIGHEST_PROTOCOL,
        )

        print(
            f"Saved intensity statistics to {f}"
        )

    resolution_dict = {}

    for mode, split_ids in zip(
        ["train", "val"],
        [training_ids, validation_ids],
    ):

        dest_mode = dest_path / mode

        print(
            f"Slicing {len(split_ids)} pairs "
            f"to {dest_mode}"
        )

        pfun = partial(
            slice_patient,
            dest_path=dest_mode,
            source_path=src_path,
            shape=tuple(args.shape),
            use_body_mask=args.body_mask,
            body_threshold=args.body_threshold,
            intensity_stats=intensity_stats,
            test_mode=False,
        )
        iterator = tqdm_(split_ids)

        match args.process:
            case 1:
                resolutions = list(
                    map(pfun, iterator)
                )

            case -1:
                resolutions = Pool().map(
                    pfun,
                    iterator,
                )

            case _ as p:
                resolutions = Pool(p).map(
                    pfun,
                    iterator,
                )

        for key, result in zip(
                split_ids,
                resolutions,
        ):
            dx, dy, dz = result

            resolution_dict[key] = (dx, dy, dz)

    with open(
        dest_path / "spacing.pkl",
        "wb",
    ) as f:

        pickle.dump(
            resolution_dict,
            f,
            pickle.HIGHEST_PROTOCOL,
        )

        print(
            f"Saved spacing dictionnary to {f}"
        )


def get_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="SegTHOR preprocessing pipeline"
    )

    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--dest_dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--shape",
        type=int,
        nargs="+",
        default=[256, 256],
    )

    parser.add_argument(
        "--retains",
        type=int,
        default=5,
        help=(
            "Number of retained patients "
            "for the validation data"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--fold",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--process",
        "-p",
        type=int,
        default=1,
        help="The number of cores to use for processing",
    )

    # -----------------------------------------------------------------
    # Stage 1 arguments
    # -----------------------------------------------------------------

    parser.add_argument(
        "--body_mask",
        action="store_true",
        help="Generate a body mask.",
    )


    parser.add_argument(
        "--body_threshold",
        type=float,
        default=-500.0,
        help=(
            "Threshold used by the body-mask method."
        ),
    )


    # -----------------------------------------------------------------
    # Stage 2 arguments
    # -----------------------------------------------------------------

    parser.add_argument(
        "--samples_per_patient",
        type=int,
        default=20000,
        help=(
            "Maximum number of body voxels sampled per training "
            "patient when fitting intensity statistics."
        ),
    )

    parser.add_argument(
        "--robust_low",
        type=float,
        default=0.5,
        help="Lower percentile for robust intensity clipping.",
    )

    parser.add_argument(
        "--robust_high",
        type=float,
        default=99.5,
        help="Upper percentile for robust intensity clipping.",
    )


    args = parser.parse_args()

    random.seed(args.seed)

    print(args)

    return args


if __name__ == "__main__":
    main(get_args())