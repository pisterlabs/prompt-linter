"""
Image transforms not provided by TensorFlow.
"""


from functools import partial
from typing import Any, Tuple, Union

import jax
import tensorflow as tf

from .constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD


PyTree = Any


def identity(input: Any, *args, **kwargs) -> Any:
    """
    Identity function that returns the input as-is.

    Args:
        input: Input that is returned as-is.
        *args: Additional positional arguments (ignored).
        **kwargs: Additional keyword arguments (ignored).

    Returns:
        Input as-is.
    """
    return input


def shape(image: tf.Tensor) -> Tuple[int, int]:
    """
    Gets the height and width of the input image.

    Args:
        image: Input image of shape [..., height, width, channels].

    Returns:
        Heigh and width of the input image.
    """
    shape_ = tf.shape(image)
    return int(shape_[-3]), int(shape_[-2])


def random_resized_crop(
    bytes: tf.Tensor,
    size: int = 224,
    method: str = 'bicubic',
    ratio: Tuple[float, float] = (3/4, 4/3),
    scale: Tuple[float, float] = (0.9, 1.0),
    antialias: bool = True,
    ) -> tf.Tensor:
    """
    Extracts a random crop from the input and resizes it to the desired size.
    Inspired by torchvision's RandomResizedCrop.

    Args:
        bytes: JPEG-encoded bytes of the input image.
        size: Size to which the random crop is resized and returned.
        method: Method to use to resize the random crop. See
            tf.image.ResizeMethod for available options.
        ratio: The lower and upper bounds for the aspect ratio of the
            random crop.
        scale: The lower and upper bounds for the area of the random crop,
            with respect to the area of the input image.
        antialias: Whether to anti-alias when resizing the random crop.

    Returns:
        A random crop of the input image resized to the desired size.
    """
    crop_top_left, crop_size, _ = tf.image.sample_distorted_bounding_box(
        image_size=tf.io.extract_jpeg_shape(bytes),
        bounding_boxes=tf.constant([0., 0., 1., 1.], shape=[1, 1, 4]),
        aspect_ratio_range=ratio,
        area_range=scale,
        max_attempts=5,
        )
    crop_top_left_y, crop_top_left_x, _ = tf.unstack(crop_top_left)
    crop_height, crop_width, _ = tf.unstack(crop_size)
    crop = tf.stack([crop_top_left_y, crop_top_left_x, crop_height, crop_width])

    cropped = tf.io.decode_and_crop_jpeg(bytes, crop, channels=3)
    resized = tf.image.resize(cropped, [size, size], method, antialias=antialias)
    return resized


def resize_smallest_edge(
    image: tf.Tensor,
    size: int = 224,
    method: str = 'bicubic',
    antialias: bool = True,
    ) -> tf.Tensor:
    """
    Resizes an image so the smallest edge is resized to the desired size and
    the aspect ratio is maintained. Inspired by torchvision's Resize.

    Args:
        image: Image to resize.
        size: Size to which the smallest edge of the input image is resized.
        method: Resizing method to use. See tf.image.ResizeMethod for available
            options.
        antialias: Whether to anti-alias when resizing.

    Returns:
        Input image with its smallest edge resized to the desired size and its
        aspect ratio maintained.
    """
    image_w, image_h = shape(image)

    if image_w <= image_h:
        size = (int(size * image_h / image_w), size)

    else:
        size = (size, int(size * image_w / image_h))

    return tf.image.resize(image, size, method, antialias=antialias)


def center_crop_with_padding(
    image: tf.Tensor,
    size: int = 224,
    ) -> tf.Tensor:
    """
    Extracts a central crop of the desired size from the input image. If the
    input image is smaller than the desired size, it is padded first. Inspired
    by torchvision's CenterCrop.

    Args:
        image: Image to center-crop.
        size: Desired size of the center-crop.

    Returns:
        Center-crop of the input image, padded if the input image is smaller
        than the desired size.
    """
    image_h, image_w = shape(image)
    padded = tf.image.pad_to_bounding_box(
        image=image,
        offset_height=tf.maximum(0, (size - image_h) // 2),
        offset_width=tf.maximum(0, (size - image_w) // 2),
        target_height=tf.maximum(size, image_h),
        target_width=tf.maximum(size, image_w),
        )

    padded_h, padded_w = shape(padded)
    cropped = tf.image.crop_to_bounding_box(
        image=padded,
        offset_height=(padded_h - size) // 2,
        offset_width=(padded_w - size) // 2,
        target_height=size,
        target_width=size,
        )

    return cropped


def normalize(
    image: tf.Tensor,
    mean: Tuple[float, ...] = OPENAI_DATASET_MEAN,
    std: Tuple[float, ...] = OPENAI_DATASET_STD,
    ) -> tf.Tensor:
    """
    Normalizes the input image at the last axis.

    Args:
        image: Image to normalize.
        mean: Mean per channel used for normalization.
        std: Standard deviation per channel used for normalization.

    Returns:
        Input image normalized using mean and std at the last axis.
    """
    image -= 255*tf.constant(mean, shape=[1, 1, 3], dtype=image.dtype)
    image /= 255*tf.constant(std, shape=[1, 1, 3], dtype=image.dtype)
    return image


def denormalize(
    image: tf.Tensor,
    mean: Tuple[float, ...] = OPENAI_DATASET_MEAN,
    std: Tuple[float, ...] = OPENAI_DATASET_STD,
    ) -> tf.Tensor:
    """
    De-normalizes the normalized input image at the last axis.

    Args:
        image: Normalized image to de-normalizes.
        mean: Mean per channel image was normalized by.
        std: Standard deviation per channel image was normalized by.

    Returns:
        Input image de-normalized at the last axis.
    """
    image *= tf.constant(std, shape=[1, 1, 3], dtype=image.dtype)
    image += tf.constant(mean, shape=[1, 1, 3], dtype=image.dtype)
    return image


def tf_to_np(pytree: PyTree, device_axis: bool = True) -> PyTree:
    """
    Converts TensorFlow tensors into NumPy arrays.

    Args:
        pytree: PyTree with TensorFlow tensors as leaves.
        device_axis: Whether to add a leading device axis to the data for
            distributed training.

    Returns:
        Input PyTree with its leaves converted into NumPy arrays and
        potentially an additional device axis.
    """
    device_count = jax.local_device_count()

    def _tf_to_jax(leaf):
        leaf = leaf._numpy()

        if device_axis:
            # [global_batch_size, ...] to [device_count, local_batch_size, ...]
            leaf = leaf.reshape((device_count, -1) + leaf.shape[1:])

        return leaf

    return jax.tree_util.tree_map(_tf_to_jax, pytree)


class Sequential:
    """
    Sequential layer that chains together a series of modules.

    Attributes:
        modules: Modules that are chained together.
    """
    def __init__(self, *modules) -> None:
        """
        Stores the modules.

        Args:
            *modules: Modules that are chained together.
        """
        self.modules = [*modules]

    def __repr__(self) -> str:
        return ' --->\n'.join(map(str, self.modules))

    def __call__(self, input: Any) -> Any:
        """
        Sequentially transforms the input by the modules.

        Args:
            input: Input that is transformed by the modules.

        Returns:
            Input sequentially transformed by the modules.
        """
        for mod in self.modules:
            input = mod(input)
        return input


def create_image_transforms(
    train: bool,
    input_format: str = 'path',
    do_batch_transforms: bool = True,
    size: int = 224,
    dtype: tf.DType = tf.float32,
    ) -> Union[Sequential, Tuple[Sequential, Sequential]]:
    """
    Creates image transforms for training or validation of CLIP models.

    Args:
        train: Whether to apply training transforms (True) or validation
            transforms (False).
        input_format: Format of the input the transforms will receive. Available
            options are 'path' for paths to images that should be read, 'bytes'
            for JPEG-encoded bytes, and 'image' for PIL images, decoded NumPy
            arrays, etc. Format 'image' is not supported for training.
        do_batch_transforms: Whether to return two separate series of transforms,
            one to be applied over individual data points and the other over batches
            for greater efficiency. If False, the two are concatenated and returned
            as one.
        size: Size to which the images are resized.
        dtype: The data type the images are converted to.

    Returns:
        If do_batch_transforms is True, two Sequential modules are returned,
        corresponding to item and batch transforms respectively. Otherwise,
        a single Sequential module is returned containing both item and
        batch transforms.

    Raises:
        ValueError: Input format is not recognized or is 'image' for training.
    """
    if input_format not in ['path', 'bytes', 'image']:
        raise ValueError(
            f'Input format {input_format} not recognized. '
            'Available options are [path, bytes, image].'
            )

    if train:
        # 'image' format is not supported for training because random resized
        # cropping expects bytes.
        if input_format == 'image':
            raise ValueError('Input format images not supported for training.')

        # The only training transform is random resized cropping.
        item_transforms = [
            tf.io.read_file if input_format == 'path' else identity,
            partial(random_resized_crop, size=size),
            ]

    else:
        # Validation transforms are resizing the smallest edge and center-cropping.
        item_transforms = [
            tf.io.read_file if input_format == 'path' else identity,
            partial(tf.io.decode_jpeg, channels=3) if input_format != 'image' else identity,
            partial(resize_smallest_edge, size=size),
            partial(center_crop_with_padding, size=size),
            ]

    # Both training and validation inputs are normalized and
    # converted to the correct data type.
    batch_transforms = [
        normalize,
        partial(tf.image.convert_image_dtype, dtype=dtype)
        ]

    if do_batch_transforms:
        transforms = (
            Sequential(*item_transforms),
            Sequential(*batch_transforms),
            )

    else:
        transforms = Sequential(*item_transforms, *batch_transforms)

    return transforms
