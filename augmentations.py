# augmentations.py

import random

import numpy as np
from PIL import Image


class IntensityAugmentation:
    """
    Online intensity augmentation for CT images.

    The same sampled augmentation parameters can be reused for
    multiple images, which is required for 2.5D input stacks.

    Augmentations:
        1. Gaussian noise
        2. Simulated low resolution
        3. Intensity scaling
    """

    def __init__(
        self,
        noise_prob=0.5,
        noise_std=0.03,
        lowres_prob=0.3,
        lowres_scale_range=(0.5, 0.75),
        intensity_prob=0.5,
        intensity_scale_range=(0.8, 1.2),
    ):
        self.noise_prob = noise_prob
        self.noise_std = noise_std

        self.lowres_prob = lowres_prob
        self.lowres_scale_range = lowres_scale_range

        self.intensity_prob = intensity_prob
        self.intensity_scale_range = intensity_scale_range

    # ------------------------------------------------------------------
    # Random parameter sampling
    # ------------------------------------------------------------------

    def sample_params(self):
        """
        Sample one set of random augmentation parameters.

        The returned parameters can be reused for multiple images.
        This is important for 2.5D, where all five neighbouring
        slices must receive the same augmentation configuration.
        """

        apply_noise = random.random() < self.noise_prob
        apply_lowres = random.random() < self.lowres_prob
        apply_intensity = random.random() < self.intensity_prob

        params = {
            "apply_noise": apply_noise,
            "apply_lowres": apply_lowres,
            "apply_intensity": apply_intensity,
            "lowres_scale": (
                random.uniform(*self.lowres_scale_range)
                if apply_lowres
                else None
            ),
            "intensity_scale": (
                random.uniform(*self.intensity_scale_range)
                if apply_intensity
                else None
            ),
        }

        return params

    # ------------------------------------------------------------------
    # Apply previously sampled parameters
    # ------------------------------------------------------------------

    def apply(self, image, params):
        """
        Apply previously sampled augmentation parameters to an image.

        Parameters
        ----------
        image : PIL.Image or np.ndarray
            Single-channel CT image.

        params : dict
            Parameters returned by sample_params().
        """

        image = self._to_float(image)

        if params["apply_noise"]:
            image = self._add_gaussian_noise(image)

        if params["apply_lowres"]:
            image = self._simulate_low_resolution(
                image,
                scale=params["lowres_scale"],
            )

        if params["apply_intensity"]:
            image = self._scale_intensity(
                image,
                scale=params["intensity_scale"],
            )

        # Keep image values valid.
        image = np.clip(image, 0.0, 1.0)

        return Image.fromarray(
            (image * 255).astype(np.uint8),
            mode="L",
        )

    # ------------------------------------------------------------------
    # Backwards-compatible 2D interface
    # ------------------------------------------------------------------

    def __call__(self, image):
        """
        Sample and apply one augmentation to a single image.

        This preserves the original 2D behaviour.
        """

        params = self.sample_params()

        return self.apply(
            image,
            params,
        )

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    @staticmethod
    def _to_float(image):
        """
        Convert an image to float32 with values in [0, 1].
        """

        if isinstance(image, Image.Image):
            image = np.asarray(image)

        image = image.astype(np.float32)

        if image.max() > 1.0:
            image /= 255.0

        return image

    def _add_gaussian_noise(self, image):
        """
        Add intensity-dependent Gaussian noise.

        Noise is sampled independently for each image/slice.
        """

        noise_scale = (
            0.005
            + self.noise_std * image
        )

        noise = np.random.normal(
            loc=0.0,
            scale=noise_scale,
            size=image.shape,
        ).astype(np.float32)

        return image + noise

    def _simulate_low_resolution(
        self,
        image,
        scale=None,
    ):
        """
        Simulate reduced spatial resolution.

        If scale is provided, it is used directly.
        If not, a random scale is sampled.

        The optional random sampling keeps this helper backwards
        compatible with direct calls.
        """

        if scale is None:
            scale = random.uniform(*self.lowres_scale_range)

        height, width = image.shape[:2]

        new_height = max(
            1,
            int(height * scale)
        )
        new_width = max(
            1,
            int(width * scale)
        )

        image_pil = Image.fromarray(
            np.clip(
                image * 255,
                0,
                255
            ).astype(np.uint8),
            mode="L",
        )

        # Downsample.
        image_pil = image_pil.resize(
            (new_width, new_height),
            resample=Image.BILINEAR,
        )

        # Upsample back to original resolution.
        image_pil = image_pil.resize(
            (width, height),
            resample=Image.BILINEAR,
        )

        return (
            np.asarray(image_pil)
            .astype(np.float32)
            / 255.0
        )

    def _scale_intensity(
        self,
        image,
        scale=None,
    ):
        """
        Randomly increase or decrease image intensity.

        If scale is provided, it is used directly.
        If not, a random scale is sampled.
        """

        if scale is None:
            scale = random.uniform(
                *self.intensity_scale_range
            )

        return image * scale


class GeometricAugmentation:
    """
    Online geometric augmentation.

    A single set of spatial parameters can be sampled and then
    applied consistently to multiple images and masks.

    This allows the same geometric transformation to be applied
    to all five slices of a 2.5D input stack and to the center GT.
    """

    def __init__(
        self,
        rotation_prob=0.5,
        rotation_range=(-15, 15),
        scaling_prob=0.5,
        scaling_range=(0.9, 1.1),
        translation_prob=0.5,
        translation_range=(-0.1, 0.1),
    ):
        self.rotation_prob = rotation_prob
        self.rotation_range = rotation_range

        self.scaling_prob = scaling_prob
        self.scaling_range = scaling_range

        self.translation_prob = translation_prob
        self.translation_range = translation_range

    # ------------------------------------------------------------------
    # Random parameter sampling
    # ------------------------------------------------------------------

    def sample_params(self):
        """
        Sample one set of random geometric parameters.

        These parameters should be reused for every slice in a
        2.5D stack and for the corresponding center GT.
        """

        apply_rotation = (
            random.random()
            < self.rotation_prob
        )

        apply_scaling = (
            random.random()
            < self.scaling_prob
        )

        apply_translation = (
            random.random()
            < self.translation_prob
        )

        params = {
            "apply_rotation": apply_rotation,
            "angle": (
                random.uniform(*self.rotation_range)
                if apply_rotation
                else None
            ),

            "apply_scaling": apply_scaling,
            "scale": (
                random.uniform(*self.scaling_range)
                if apply_scaling
                else None
            ),

            "apply_translation": apply_translation,
            "tx": (
                random.uniform(*self.translation_range)
                if apply_translation
                else None
            ),
            "ty": (
                random.uniform(*self.translation_range)
                if apply_translation
                else None
            ),
        }

        return params

    # ------------------------------------------------------------------
    # Apply previously sampled parameters
    # ------------------------------------------------------------------

    def apply(
        self,
        image,
        params,
        is_mask=False,
    ):
        """
        Apply previously sampled geometric parameters.

        Parameters
        ----------
        image : PIL.Image or np.ndarray
            Image or segmentation mask.

        params : dict
            Parameters returned by sample_params().

        is_mask : bool
            True for GT masks, false for CT images.

        Notes
        -----
        CT images use bilinear interpolation.
        GT masks use nearest-neighbour interpolation.
        """

        image = self._to_pil(image)

        resample = (
            Image.NEAREST
            if is_mask
            else Image.BILINEAR
        )

        # --------------------------------------------------------------
        # Rotation
        # --------------------------------------------------------------

        if params["apply_rotation"]:
            image = image.rotate(
                params["angle"],
                resample=resample,
                fillcolor=0,
            )

        # --------------------------------------------------------------
        # Scaling
        # --------------------------------------------------------------

        if params["apply_scaling"]:
            image = self._scale(
                image,
                params["scale"],
                resample=resample,
            )

        # --------------------------------------------------------------
        # Translation
        # --------------------------------------------------------------

        if params["apply_translation"]:
            image = self._translate(
                image,
                params["tx"],
                params["ty"],
                resample=resample,
            )

        return image

    # ------------------------------------------------------------------
    # Backwards-compatible 2D interface
    # ------------------------------------------------------------------

    def __call__(self, image, gt):
        """
        Sample one geometric transform and apply it to both image
        and GT.

        This preserves the original 2D behaviour.
        """

        params = self.sample_params()

        image = self.apply(
            image,
            params,
            is_mask=False,
        )

        gt = self.apply(
            gt,
            params,
            is_mask=True,
        )

        return image, gt

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    @staticmethod
    def _to_pil(image):
        if isinstance(image, Image.Image):
            return image

        return Image.fromarray(
            np.asarray(image)
        )

    @staticmethod
    def _scale(
        image,
        scale,
        resample,
    ):
        width, height = image.size

        new_width = max(
            1,
            int(width * scale)
        )

        new_height = max(
            1,
            int(height * scale)
        )

        scaled = image.resize(
            (new_width, new_height),
            resample=resample,
        )

        # --------------------------------------------------------------
        # Scaling up: crop the center
        # --------------------------------------------------------------

        if scale >= 1:
            left = (
                new_width - width
            ) // 2

            top = (
                new_height - height
            ) // 2

            return scaled.crop(
                (
                    left,
                    top,
                    left + width,
                    top + height,
                )
            )

        # --------------------------------------------------------------
        # Scaling down: paste centered on zero canvas
        # --------------------------------------------------------------

        canvas = Image.new(
            image.mode,
            (width, height),
            0,
        )

        left = (
            width - new_width
        ) // 2

        top = (
            height - new_height
        ) // 2

        canvas.paste(
            scaled,
            (left, top),
        )

        return canvas

    @staticmethod
    def _translate(
        image,
        tx,
        ty,
        resample,
    ):
        width, height = image.size

        shift_x = int(tx * width)
        shift_y = int(ty * height)

        return image.transform(
            (width, height),
            Image.AFFINE,
            (
                1,
                0,
                -shift_x,
                0,
                1,
                -shift_y,
            ),
            resample=resample,
            fillcolor=0,
        )