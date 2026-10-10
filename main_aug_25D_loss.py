#!/usr/bin/env python3

# MIT License

# Copyright (c) 2025 Hoel Kervadec, Caroline Magg

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

import argparse
import warnings
from typing import Any
from pathlib import Path
from pprint import pprint
from operator import itemgetter
from shutil import copytree, rmtree
import re
from PIL import Image

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision import transforms
from torch.utils.data import DataLoader

from functools import partial

from dataset import SliceDataset as SliceDataset2D
from dataset_25D import SliceDataset as SliceDataset25D

from ShallowNet import shallowCNN
from ENet import ENet as ENet2D
from ENet_25D import ENet as ENet25D
from UNetPlusPlus import UNetPlusPlus

from augmentations import GeometricAugmentation, IntensityAugmentation

from utils import (Dcm,
                   class2one_hot,
                   probs2one_hot,
                   probs2class,
                   tqdm_,
                   dice_coef,
                   save_images)

from losses import (CrossEntropy, WeightedCrossEntropy, DiceLoss, CEDice)



#!!! Inverse-square-root class-frequency weighting,
# calculated from pixel frequencies in the corrected SEGTHOR training set.
class_weights = torch.tensor([
    0.0535,   # class 0: background
    2.4610,   # class 1: esophagus
    0.1897,   # class 2: heart
    2.8586,   # class 3: trachea
    0.3859    # class 4: aorta
], dtype=torch.float32) ###!!!

datasets_params: dict[str, dict[str, Any]] = {}

datasets_params["TOY2"] = {
    'K': 2,
    'B': 2,
    'kernels': 8,
    'factor': 2
}

datasets_params["SEGTHOR"] = {
    'K': 5,
    'B': 8,
    'kernels': 8,
    'factor': 2
}

datasets_params["SEGTHOR_CLEAN"] = {
    'K': 5,
    'B': 8,
    'kernels': 8,
    'factor': 2
}

datasets_params["SEGTHOR_CLEAN_FINAL"] = {
    'K': 5,
    'B': 8,
    'kernels': 8,
    'factor': 2
}

datasets_params["SEGTHOR_FULL_SLICED"] = {
    'K': 5,
    'B': 8,
    'kernels': 8,
    'factor': 2
}

datasets_params["SEGTHOR_FULL_PREPROCESSED"] = {
    'K': 5,
    'B': 8,
    'kernels': 8,
    'factor': 2
}



def img_transform(img):
        img = img.convert('L')
        img = np.array(img)[np.newaxis, ...]
        img = img / 255  # max <= 1
        img = torch.tensor(img, dtype=torch.float32)
        return img

def gt_transform(K, img):
        img = np.array(img)[...]
        # The idea is that the classes are mapped to {0, 255} for binary cases
        # {0, 85, 170, 255} for 4 classes
        # {0, 51, 102, 153, 204, 255} for 6 classes
        # Very sketchy but that works here and that simplifies visualization
        img = img / (255 / (K - 1)) if K != 5 else img / 63  # max <= 1
        img = torch.tensor(img, dtype=torch.int64)[None, ...]  # Add one dimension to simulate batch
        img = class2one_hot(img, K=K)
        return img[0]

def setup(args) -> tuple[nn.Module, Any, Any, DataLoader, DataLoader, int]:
    # Device
    gpu: bool = args.gpu and torch.cuda.is_available()
    device = torch.device("cuda") if gpu else torch.device("cpu")
    print(f">> Picked {device} to run experiments")

    # Dataset parameters
    K: int = datasets_params[args.dataset]['K']
    B: int = datasets_params[args.dataset]['B']
    kernels: int = datasets_params[args.dataset].get('kernels', 8)
    factor: int = datasets_params[args.dataset].get('factor', 2)

    root_dir = Path("data") / args.dataset

    # Choose architecture + dataset loader

    if args.architecture == "2d":

        if args.dataset == "TOY2":
            dataset_class = SliceDataset2D
            in_channels = 1
            net_class = shallowCNN if args.net == "enet" else UNetPlusPlus
        else:
            dataset_class = SliceDataset2D
            in_channels = 1
            net_class = ENet2D if args.net == "enet" else UNetPlusPlus

    elif args.architecture == "2.5d":

        if args.dataset == "TOY2":
            raise ValueError("2.5D architecture is only intended for SEGTHOR datasets.")

        dataset_class = SliceDataset25D
        in_channels = 5
        net_class = ENet25D if args.net == "enet" else UNetPlusPlus

    else:
        raise ValueError(args.architecture)

    # Build network
    if args.net == "unetpp":
        net = UNetPlusPlus(
            in_channels,
            K,
            kernels=32
        )
    else:
        net = net_class(
            in_channels,
            K,
            kernels=kernels,
            factor=factor
        )

    net.init_weights()
    net.to(device)

    print(
        f">> Architecture: {args.architecture}, "
        f"in_channels={in_channels}"
    )

    # Optimizer
    lr = 0.0005

    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=lr,
        betas=(0.9, 0.999)
    )

   #Augmentations
    intensity_augmentation = None
    geometric_augmentation = None

    if args.intensity_aug:
        intensity_augmentation = IntensityAugmentation(
            noise_prob=0.5,
            noise_std=0.03,
            lowres_prob=0.3,
            lowres_scale_range=(0.5, 0.75),
            intensity_prob=0.5,
            intensity_scale_range=(0.8, 1.2),
        )

    if args.geometric_aug:
        geometric_augmentation = GeometricAugmentation(
            rotation_prob=0.5,
            rotation_range=(-30, 30),
            scaling_prob=0.5,
            scaling_range=(0.8, 1.2),
            translation_prob=0.5,
            translation_range=(-0.2, 0.2),
        )

    # Training dataset
    if args.architecture == "2d":

        train_set = dataset_class(
            'train',
            root_dir,
            img_transform=img_transform,
            gt_transform=partial(gt_transform, K),
            augment=intensity_augmentation,
            geometric_augment=geometric_augmentation,
            debug=args.debug,
        )


    else:
        train_set = dataset_class(
            'train',
            root_dir,
            img_transform=img_transform,
            gt_transform=partial(gt_transform, K),
            augment=intensity_augmentation,
            geometric_augment=geometric_augmentation,
            debug=args.debug,
        )

    train_loader = DataLoader(
        train_set,
        batch_size=B,
        num_workers=5,
        shuffle=True
    )

    # Validation dataset (no aug)
    val_set = dataset_class(
        'val',
        root_dir,
        img_transform=img_transform,
        gt_transform=partial(gt_transform, K),
        debug=args.debug,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=B,
        num_workers=5,
        shuffle=False
    )

    print(f">> root={root_dir}")
    print(f">> Architecture={args.architecture}")
    print(f">> Intensity augmentation={args.intensity_aug}")
    print(f">> Geometric augmentation={args.geometric_aug}")

    args.dest.mkdir(parents=True, exist_ok=True)

    return (
        net,
        optimizer,
        device,
        train_loader,
        val_loader,
        K
    )


def setup_loss(args, K: int):
    if args.mode == "full":
        idk = list(range(K))  # Supervise both background and foreground
    elif args.mode in ["partial"] and args.dataset == 'SEGTHOR':
        return CrossEntropy(idk=[0, 1, 3, 4])  # Do not supervise the heart (class 2)
    else:
        raise ValueError(args.mode, args.dataset)

    # The background is ~99% of the pixels and sits at a permanent DSC of .998,
    # so it is kept out of any Dice term: averaged in, it would only dilute the
    # organs that term exists to rebalance. The cross-entropy still supervises it.
    foreground = [k for k in idk if k != 0]

    match args.loss:
        case 'ce':
            return CrossEntropy(idk=idk)
        case 'wce':
            # The weighted CE experiment, reachable through --loss instead of
            # by uncommenting it here
            return WeightedCrossEntropy(idk=idk, weights=class_weights)
        case 'dice':
            return DiceLoss(idk=foreground)
        case 'cedice':
            return CEDice(idk=idk)
        case _:
            raise ValueError(args.loss)


def runTraining(args):
    print(f">>> Setting up to train on {args.dataset} with {args.mode} and {args.loss}")
    net, optimizer, device, train_loader, val_loader, K = setup(args)

    loss_fn = setup_loss(args, K)

    # Notice one has the length of the _loader_, and the other one of the _dataset_
    log_loss_tra: Tensor = torch.zeros((args.epochs, len(train_loader)))
    log_dice_tra: Tensor = torch.zeros((args.epochs, len(train_loader.dataset), K))
    log_loss_val: Tensor = torch.zeros((args.epochs, len(val_loader)))
    log_dice_val: Tensor = torch.zeros((args.epochs, len(val_loader.dataset), K))

    best_dice: float = 0

    for e in range(args.epochs):
        for m in ['train', 'val']:
            match m:
                case 'train':
                    net.train()
                    opt = optimizer
                    cm = Dcm
                    desc = f">> Training   ({e: 4d})"
                    loader = train_loader
                    log_loss = log_loss_tra
                    log_dice = log_dice_tra
                case 'val':
                    net.eval()
                    opt = None
                    cm = torch.no_grad
                    desc = f">> Validation ({e: 4d})"
                    loader = val_loader
                    log_loss = log_loss_val
                    log_dice = log_dice_val

            with cm():  # Either dummy context manager, or the torch.no_grad for validation
                j = 0
                tq_iter = tqdm_(enumerate(loader), total=len(loader), desc=desc)
                for i, data in tq_iter:
                    img = data['images'].to(device)
                    gt = data['gts'].to(device)

                    if opt:  # So only for training
                        opt.zero_grad()

                    # Sanity tests to see we loaded and encoded the data correctly
                    assert 0 <= img.min() and img.max() <= 1
                    B, _, W, H = img.shape

                    pred_logits = net(img)
                    pred_probs = F.softmax(1 * pred_logits, dim=1)  # 1 is the temperature parameter

                    # Metrics computation, not used for training
                    pred_seg = probs2one_hot(pred_probs)
                    log_dice[e, j:j + B, :] = dice_coef(pred_seg, gt)  # One DSC value per sample and per class

                    loss = loss_fn(pred_probs, gt)
                    log_loss[e, i] = loss.item()  # One loss value per batch (averaged in the loss)

                    if opt:  # Only for training
                        loss.backward()
                        opt.step()

                    if m == 'val':
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', category=UserWarning)
                            predicted_class: Tensor = probs2class(pred_probs)
                            mult: int = 63 if K == 5 else (255 / (K - 1))
                            save_images(predicted_class * mult,
                                        data['stems'],
                                        args.dest / f"iter{e:03d}" / m)

                    j += B  # Keep in mind that _in theory_, each batch might have a different size
                    # For the DSC average: do not take the background class (0) into account:
                    postfix_dict: dict[str, str] = {"Dice": f"{log_dice[e, :j, 1:].mean():05.3f}",
                                                    "Loss": f"{log_loss[e, :i + 1].mean():5.2e}"}
                    if K > 2:
                        postfix_dict |= {f"Dice-{k}": f"{log_dice[e, :j, k].mean():05.3f}"
                                         for k in range(1, K)}
                    tq_iter.set_postfix(postfix_dict)

        # I save it at each epochs, in case the code crashes or I decide to stop it early
        np.save(args.dest / "loss_tra.npy", log_loss_tra)
        np.save(args.dest / "dice_tra.npy", log_dice_tra)
        np.save(args.dest / "loss_val.npy", log_loss_val)
        np.save(args.dest / "dice_val.npy", log_dice_val)

        current_dice: float = log_dice_val[e, :, 1:].mean().item()
        if current_dice > best_dice:
            message = f">>> Improved dice at epoch {e}: {best_dice:05.3f}->{current_dice:05.3f} DSC"
            print(message)
            best_dice = current_dice
            with open(args.dest / "best_epoch.txt", 'w') as f:
                f.write(message)

            best_folder = args.dest / "best_epoch"
            if best_folder.exists():
                rmtree(best_folder)
            copytree(args.dest / f"iter{e:03d}", Path(best_folder))

            torch.save(net, args.dest / "bestmodel.pkl")
            torch.save(net.state_dict(), args.dest / "bestweights.pt")


def runInference(args):
    """Run 2.5D inference on the preprocessed test PNG slices."""

    if args.architecture != "2.5d":
        raise ValueError("Inference requires --architecture 2.5d.")
    if args.net != "unetpp":
        raise ValueError("Inference requires --net unetpp.")

    K = datasets_params[args.dataset]["K"]
    image_dir = Path("data") / args.dataset / "test" / "img"
    checkpoint_path = args.checkpoint or (args.dest / "bestweights.pt")
    output_dir = args.dest / "test_pred_png"

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Test images not found: {image_dir}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if output_dir.exists() and any(output_dir.glob("*.png")):
        raise FileExistsError(
            f"Predictions already exist in {output_dir}. "
            "Move or remove the folder before rerunning."
        )

    # Group slices by patient.
    patient_slices = {}
    for path in image_dir.glob("Patient_*.png"):
        match = re.fullmatch(r"(Patient_\d+)_([0-9]+)\.png", path.name)
        if match is None:
            raise ValueError(f"Unexpected slice filename: {path.name}")
        patient_id, z = match.group(1), int(match.group(2))
        patient_slices.setdefault(patient_id, []).append((z, path))

    if not patient_slices:
        raise FileNotFoundError(f"No test PNG slices found in {image_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> Inference device: {device}")
    print(f">> Loading checkpoint: {checkpoint_path}")

    # Recreate the same five-input-channel UNet++ used for training.
    net = UNetPlusPlus(5, K, kernels=32).to(device)

    try:
        weights = torch.load(
            checkpoint_path, map_location=device, weights_only=True
        )
    except TypeError:
        weights = torch.load(checkpoint_path, map_location=device)

    net.load_state_dict(weights)
    net.eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    offsets = (-2, -1, 0, 1, 2)
    total_saved = 0

    with torch.inference_mode():
        for patient_id in sorted(patient_slices):
            entries = sorted(patient_slices[patient_id], key=lambda item: item[0])

            if [z for z, _ in entries] != list(range(len(entries))):
                raise ValueError(f"Nonconsecutive slice indices for {patient_id}")

            paths = [path for _, path in entries]
            images = np.stack([
                np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
                for path in paths
            ]).astype(np.float32) / 255.0

            depth = len(paths)
            print(f">> Predicting {patient_id}: {depth} slices")

            for start in range(0, depth, args.inference_batch_size):
                stop = min(start + args.inference_batch_size, depth)
                centers = np.arange(start, stop)

                # Build [batch, 5, height, width], replicating boundary slices.
                context = np.stack([
                    images[np.clip(centers + offset, 0, depth - 1)]
                    for offset in offsets
                ], axis=1)

                x = torch.from_numpy(context).to(device=device)
                logits = net(x)

                if isinstance(logits, (tuple, list)):
                    logits = logits[-1]

                labels = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)

                for i, z in enumerate(range(start, stop)):
                    # Encode class IDs 0–4 as PNG values 0, 63, 126, 189, 252.
                    encoded = (labels[i] * 63).astype(np.uint8)
                    Image.fromarray(encoded).save(output_dir / paths[z].name)
                    total_saved += 1

    print(">> Inference complete.")
    print(f">> Saved {total_saved} PNG masks to: {output_dir}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--dataset', default='TOY2', choices=datasets_params.keys())
    parser.add_argument('--mode', default='full', choices=['partial', 'full'])
    parser.add_argument(
        '--architecture',
        default='2d',
        choices=['2d', '2.5d'],
        help="Segmentation architecture: standard 2D or 2.5D using 5 axial slices."
    )

    parser.add_argument(
        '--net',
        default='enet',
        choices=['enet', 'unetpp'],
        help="Network architecture: ENet or UNet++."
    )

    parser.add_argument(
        '--intensity_aug',
        action='store_true',
        help="Enable online intensity augmentation during training."
    )

    parser.add_argument(
        '--geometric_aug',
        action='store_true',
        help="Enable online geometric augmentation during training."
    )
    parser.add_argument('--loss', default='ce', choices=['ce', 'wce', 'dice', 'cedice'],
                        help="Training objective: cross-entropy (the baseline), class-weighted "
                             "cross-entropy, soft Dice, or an equally weighted sum of CE and Dice.")
    parser.add_argument('--dest', type=Path, required=True,
                        help="Destination directory to save the results (predictions and weights).")

    parser.add_argument('--gpu', action='store_true')

    parser.add_argument(
        '--inference',
        action='store_true',
        help='Run test inference instead of training.'
    )
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=None,
        help='Checkpoint path; defaults to <dest>/bestweights.pt.'
    )
    parser.add_argument(
        '--inference_batch_size',
        type=int,
        default=8,
        help='Number of test slices per inference batch.'
    )

    parser.add_argument('--debug', action='store_true',
                        help="Keep only a fraction (10 samples) of the datasets, "
                             "to test the logics around epochs and logging easily.")


    args = parser.parse_args()

    pprint(args)

    if args.inference:
        if args.inference_batch_size < 1:
            parser.error("--inference_batch_size must be at least 1")
        runInference(args)
    else:
        runTraining(args)



if __name__ == '__main__':
    main()
