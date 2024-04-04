import math
import numpy as np


def _pad_image(image: np.ndarray, pad_widths: tuple[int, int], axis: str) -> np.ndarray:
    if axis == 'x':
        axis_pad = ((0, 0), (0, 0), pad_widths)
    elif axis == 'y':
        axis_pad = ((0, 0), pad_widths, (0, 0))
    else:
        raise ValueError('axis should be either x or y')

    return np.pad(image.copy(), axis_pad, mode='edge')


def _crop_image(image: np.ndarray, crop_widths: tuple[int, int], axis: str) -> np.ndarray:
    if axis == 'x':
        return image[:, :, crop_widths[0]:-crop_widths[1]]
    elif axis == 'y':
        return image[:, crop_widths[0]:-crop_widths[1], :]
    else:
        raise ValueError('axis should be either x or y')


def _calc_pad_widths(init_length: int, dist_length: int) -> tuple[int, int]:
    return (
        int(math.floor((dist_length - init_length) / 2)),
        # pad 1 px more at the end of the dimension in case the total
        # number of pixels to pad is odd
        int(math.ceil((dist_length - init_length) / 2)),
    )


def _calc_crop_widths(init_length: int, dist_length: int) -> tuple[int, int]:
    return (
        # cut 1 px more at the beginning of the dimension in case the total
        # number of pixels to cut is odd
        int(math.ceil((init_length - dist_length) / 2)),
        int(math.floor((init_length - dist_length) / 2)),
    )


def _resize_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    res_image = image.copy()
    image_width = image.shape[2]
    image_height = image.shape[1]

    if width > image_width:
        pad_widths = _calc_pad_widths(image_width, width)
        res_image = _pad_image(res_image, pad_widths, axis='x')
    elif width < image_width:
        crop_widths = _calc_crop_widths(image_width, width)
        res_image = _crop_image(res_image, crop_widths, axis='x')

    if height > image_height:
        pad_widths = _calc_pad_widths(image_height, height)
        res_image = _pad_image(res_image, pad_widths, axis='y')
    elif height < image_height:
        crop_widths = _calc_crop_widths(image_height, height)
        res_image = _crop_image(res_image, crop_widths, axis='y')

    return res_image


def prepare_image(image: np.ndarray,
                  width: int,
                  height: int,
                  x: int,
                  y: int,
                  size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepares the given image, represented as 3D NumPy ``ndarray``,
    for classification tasks by padding and/or cropping it to the desired
    ``width`` and ``height``. The function returns the tuple of cropped/resized
    copy of ``image`` and the subarea specified by ``x``, ``y`` and ``size``
    parameters.

    :param image: 3D NumPy ``ndarray`` with shape (1, H, W) (H and W are height
    and width, respectively) that contains a grayscale image.
    :param width: the width of the resized image.
    :param height: the height of the resized image.
    :param x: the x-coordinate within the resized image where the subarea
        should start.
    :param y: the y-coordinate within the resized image where the subarea
        should start.
    :param size: the size in both dimensions of the cropped subarea.
    :raises ValueError:
        - if ``image`` does not have exactly 3 dimensions;
        - if ``image``'s channel size is not exactly 1;
        - if ``width``, ``height`` or ``size`` are less than 32;
        - if ``x`` or ``y`` are negative;
        - if the subarea exceeds the resized image's width and height.
    :return: a tuple of:
        1.) a 3D NumPy ``ndarray`` of (1, ``height``, ``width``) shape that
            represents the resized copied version of ``image``. It has the same
            data type as ``image``.
        2.) a 3D NumPy ``ndarray`` of (1, ``size``, ``size``) shape that
            represents the subarea of ``image``. It has the same data type as
            ``image``.
    """
    if image.ndim != 3:
        raise ValueError('Image shape should be exactly 3D')
    if image.shape[0] != 1:
        raise ValueError('Channel size should be exactly 1')
    if width < 32 or height < 32 or size < 32:
        raise ValueError('width, hight and size should be >= 32')
    if x < 0 or y < 0:
        raise ValueError('x or y should be >= 0')
    if (x + size) > width or (y + size) > height:
        raise ValueError(
            'the subarea should not exceed the resized image width and height'
        )

    resized_image = _resize_image(image, width, height)
    subarea = resized_image[:, y:(y + size), x:(x + size)]

    return (resized_image, subarea)

