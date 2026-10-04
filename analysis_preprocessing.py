from pathlib import Path
import argparse

import nibabel as nib
import numpy as np


def describe_ct(ct_path: Path):
    nii = nib.load(str(ct_path))
    header = nii.header
    proxy = nii.dataobj

    # Actual values as exposed by nibabel's data proxy.
    # This includes NIfTI scaling if present.
    ct = np.asarray(proxy)

    # Raw stored values, before NIfTI scaling.
    raw = np.asarray(proxy.get_unscaled())

    print(f"\nCT: {ct_path}")
    print("-" * 80)

    print(f"shape:              {ct.shape}")
    print(f"dtype stored:       {raw.dtype}")
    print(f"dtype loaded:       {ct.dtype}")

    print(f"zooms:               {header.get_zooms()[:3]}")
    print(f"voxel count:         {ct.size:,}")

    print("\nNIfTI intensity scaling:")
    print(f"  scl_slope header:  {header['scl_slope']}")
    print(f"  scl_inter header:  {header['scl_inter']}")

    print("\nAffine:")
    print(nii.affine)

    print("\nOrientation:")
    print(nib.aff2axcodes(nii.affine))

    print("\nRaw stored-value statistics:")
    print(f"  min:               {raw.min()}")
    print(f"  max:               {raw.max()}")
    print(f"  mean:              {raw.mean():.3f}")

    print("\nLoaded-value statistics:")
    print(f"  min:               {ct.min():.3f}")
    print(f"  max:               {ct.max():.3f}")
    print(f"  mean:              {ct.mean():.3f}")
    print(f"  std:               {ct.std():.3f}")

    percentiles = [0, 0.1, 1, 5, 25, 50, 75, 95, 99, 99.9, 100]
    values = np.percentile(ct, percentiles)

    print("\nIntensity percentiles:")
    for p, value in zip(percentiles, values):
        print(f"  p{p:>5}:            {value:10.3f}")

    # Useful for seeing how much of the volume is around air values.
    for threshold in [-1000, -950, -500, -100, 0, 100, 200, 300]:
        fraction = np.mean(ct <= threshold)
        print(
            f"fraction <= {threshold:>5}: "
            f"{100.0 * fraction:7.3f}%"
        )

    return nii, ct


def describe_gt(gt_path: Path):
    nii = nib.load(str(gt_path))
    gt = np.asarray(nii.dataobj)

    print(f"\nGT: {gt_path}")
    print("-" * 80)

    print(f"shape:              {gt.shape}")
    print(f"dtype:              {gt.dtype}")
    print(f"zooms:              {nii.header.get_zooms()[:3]}")
    print(f"unique labels:      {np.unique(gt)}")

    values, counts = np.unique(gt, return_counts=True)

    print("\nGT voxel counts:")
    total = gt.size

    for value, count in zip(values, counts):
        print(
            f"  label {int(value):>3}: "
            f"{count:>12,} voxels "
            f"({100.0 * count / total:8.4f}%)"
        )

    return nii, gt


def compare_ct_gt(ct_path: Path, gt_path: Path):
    ct_nii = nib.load(str(ct_path))
    gt_nii = nib.load(str(gt_path))

    ct = np.asarray(ct_nii.dataobj)
    gt = np.asarray(gt_nii.dataobj)

    print("\nCT / GT consistency:")
    print("-" * 80)

    print(f"same shape:         {ct.shape == gt.shape}")
    print(f"CT shape:           {ct.shape}")
    print(f"GT shape:           {gt.shape}")

    print(
        "same spacing:       ",
        np.allclose(
            ct_nii.header.get_zooms()[:3],
            gt_nii.header.get_zooms()[:3],
        ),
    )

    print(
        "same affine:        ",
        np.allclose(ct_nii.affine, gt_nii.affine),
    )


def inspect_patient(patient_dir: Path):
    patient_id = patient_dir.name

    ct_path = patient_dir / f"{patient_id}.nii.gz"
    gt_path = patient_dir / "GT.nii.gz"

    if not ct_path.exists():
        print(f"\nWARNING: missing CT: {ct_path}")
        return

    describe_ct(ct_path)

    if gt_path.exists():
        describe_gt(gt_path)
        compare_ct_gt(ct_path, gt_path)
    else:
        print(f"\nNo GT found: {gt_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect actual SegTHOR NIfTI files before preprocessing."
    )

    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data/segthor_full/train"),
        help="Directory containing Patient_XX folders.",
    )

    parser.add_argument(
        "--patients",
        nargs="+",
        default=["Patient_01"],
        help="Patient IDs to inspect.",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Inspect all Patient_* folders.",
    )

    args = parser.parse_args()

    assert args.data_dir.exists(), args.data_dir

    if args.all:
        patient_dirs = sorted(
            p for p in args.data_dir.glob("Patient_*")
            if p.is_dir()
        )
    else:
        patient_dirs = [
            args.data_dir / patient_id
            for patient_id in args.patients
        ]

    print("=" * 80)
    print("SEGTHOR NIfTI INSPECTION")
    print("=" * 80)
    print(f"data directory: {args.data_dir}")
    print(f"patients:       {[p.name for p in patient_dirs]}")
    print("=" * 80)

    for patient_dir in patient_dirs:
        if not patient_dir.exists():
            print(f"\nWARNING: patient directory does not exist: {patient_dir}")
            continue

        inspect_patient(patient_dir)


if __name__ == "__main__":
    main()