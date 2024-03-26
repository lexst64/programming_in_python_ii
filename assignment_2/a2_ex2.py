import numpy as np


def _pad_image(image: np.ndarray, pad_widths: tuple[int, int], axis: str) -> np.ndarray:
    if axis == 'x':
        # pixels to add to x axis
        axis_pad = ((0, 0), (0, 0), pad_widths)
    elif axis == 'y':
        # pixels to add to y axis
        axis_pad = ((0, 0), pad_widths, (0, 0))
    else:
        raise ValueError('axis should be either x or y')
    
    return np.pad(image.copy(), axis_pad, mode='edge')


def _crop_image(image: np.ndarray, crop_widths: tuple[int, int], axis: str) -> np.ndarray:
    if axis == 'x':
        # indexes of the elements to delete from x axis
        indexes = (
            list(range(0, crop_widths[0]))
            + list(range(image.shape[2] - crop_widths[1], image.shape[2]))
        )
    elif axis == 'y':
        # indexes of the elements to delete from y axis
        indexes = (
            list(range(0, crop_widths[0]))
            + list(range(image.shape[1] - crop_widths[1], image.shape[1]))
        )
    else:
        raise ValueError('axis should be either x or y')
    
    return np.delete(image.copy(), indexes, axis=(2 if axis == 'x' else 1))


def _calc_pad_widths(init_length: int , dist_length: int) -> tuple[int, int]:
    # additional 1px padding in case total number of pixels to add is odd
    add_pad = 1 if (dist_length - init_length) % 2 != 0 else 0
    return (
        (dist_length - init_length) // 2,
        (dist_length - init_length) // 2 + add_pad,
    )


def _calc_crop_widths(init_length: int, dist_length: int) -> tuple[int, int]:
    # 1px to subtract in case total number of pixels to subtract is odd
    subtract_crop = 1 if (init_length - dist_length) % 2 != 0 else 0
    return (
        (init_length - dist_length) // 2,
        (init_length - dist_length) // 2 + subtract_crop,
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


def _subarea_image(image: np.ndarray, x: int, y: int, size: int) -> np.ndarray:
    sub_image = image.copy()
    
    x_indexes = list(range(0, x)) + list(range(x + size + 1, image.shape[2]))
    y_indexes = list(range(0, y)) + list(range(y + size + 1, image.shape[1]))
    
    sub_image = np.delete(sub_image, x_indexes, axis=2)
    sub_image = np.delete(sub_image, y_indexes, axis=1)
    
    return sub_image


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
    :param width: the width of the resulting image.
    :param height: the height of the resulting image.
    :param x: the x-coordinate within resized_image where the subarea
        should start.
    :param y: the y-coordinate within resized_image where the subarea
        should start.
    :param size: the size in both dimensions of the cropped subarea.
    :raises: 
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
    
    return (
        _resize_image(image, width, height),
        _subarea_image(image, x, y, size)
    )

