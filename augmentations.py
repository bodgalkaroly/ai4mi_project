# augmentations.py

import random

import numpy as np
from PIL import Image


class IntensityAugmentation:
    """
    Online intensity augmentation for 2D CT images.

    The segmentation ground truth is not modified.

    Augmentations:
        1. Gaussian noise
        2. Simulated low resolution
        3. Intensity scaling

    Parameters
    ----------
    noise_prob : float
        Probability of applying Gaussian noise.
    noise_std : float
        Standard deviation of Gaussian noise, relative to an image
        represented in the [0, 1] range.

    lowres_prob : float
        Probability of applying simulated low-resolution augmentation.
    lowres_scale_range : tuple[float, float]
        Range of downsampling factors.

    intensity_prob : float
        Probability of applying intensity scaling.
    intensity_scale_range : tuple[float, float]
        Range of multiplicative intensity factors.
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

    def __call__(self, image):
        """
        Apply random intensity augmentations to an image.

        Parameters
        ----------
        image : PIL.Image or np.ndarray
            Single-channel CT image.

        Returns
        -------
        PIL.Image
            Augmented single-channel CT image.
        """

        image = self._to_float(image)

        if random.random() < self.noise_prob:
            image = self._add_gaussian_noise(image)

        if random.random() < self.lowres_prob:
            image = self._simulate_low_resolution(image)

        if random.random() < self.intensity_prob:
            image = self._scale_intensity(image)

        # Keep image values valid.
        image = np.clip(image, 0.0, 1.0)

        return Image.fromarray(
            (image * 255).astype(np.uint8),
            mode="L",
        )

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

        The first term provides a small amount of noise everywhere.
        The second term makes noise stronger in higher-intensity regions.
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

    def _simulate_low_resolution(self, image):
        """
        Simulate reduced spatial resolution.

        The image is downsampled and then upsampled back to its
        original dimensions.
        """

        scale = random.uniform(*self.lowres_scale_range)

        height, width = image.shape[:2]

        new_height = max(1, int(height * scale))
        new_width = max(1, int(width * scale))

        image_pil = Image.fromarray(
            np.clip(image * 255, 0, 255).astype(np.uint8),
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

        return np.asarray(image_pil).astype(np.float32) / 255.0

    def _scale_intensity(self, image):
        """
        Randomly increase or decrease image intensity.
        """

        scale = random.uniform(*self.intensity_scale_range)

        return image * scale

class GeometricAugmentation:
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

    def __call__(self, image):
        image = self._to_pil(image)

        if random.random() < self.rotation_prob:
            angle = random.uniform(*self.rotation_range)
            image = image.rotate(
                angle,
                resample=Image.BILINEAR,
                fillcolor=0,
            )

        if random.random() < self.scaling_prob:
            scale = random.uniform(*self.scaling_range)
            image = self._scale(image, scale)

        if random.random() < self.translation_prob:
            tx = random.uniform(*self.translation_range)
            ty = random.uniform(*self.translation_range)
            image = self._translate(image, tx, ty)

        return image

    @staticmethod
    def _to_pil(image):
        if isinstance(image, Image.Image):
            return image.convert("L")

        image = np.asarray(image)

        if image.dtype != np.uint8:
            image = np.clip(image, 0, 1)
            image = (image * 255).astype(np.uint8)

        return Image.fromarray(image, mode="L")

    @staticmethod
    def _scale(image, scale):
        width, height = image.size

        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))

        scaled = image.resize(
            (new_width, new_height),
            resample=Image.BILINEAR,
        )

        if scale >= 1:
            left = (new_width - width) // 2
            top = (new_height - height) // 2
            return scaled.crop(
                (left, top, left + width, top + height)
            )

        canvas = Image.new("L", (width, height), 0)

        left = (width - new_width) // 2
        top = (height - new_height) // 2

        canvas.paste(scaled, (left, top))

        return canvas

    @staticmethod
    def _translate(image, tx, ty):
        width, height = image.size

        shift_x = int(tx * width)
        shift_y = int(ty * height)

        return image.transform(
            (width, height),
            Image.AFFINE,
            (1, 0, -shift_x, 0, 1, -shift_y),
            resample=Image.BILINEAR,
            fillcolor=0,
        )