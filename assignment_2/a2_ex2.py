import numpy as np


def _pad_image(image: np.ndarray, pad_width: int, axis: str) -> np.ndarray:
    if axis == 'x':
        axis_pad = ((0, 0), (0, 0), (pad_width, pad_width))
    elif axis == 'y':
        axis_pad = ((0, 0), (pad_width, pad_width), (0, 0))
    else:
        raise ValueError('axis should be either x or y')
    
    return np.pad(image.copy(), axis_pad, mode='edge')


def _crop_image(image: np.ndarray, crop_width: int, axis: str) -> np.ndarray:
    if axis == 'x':
        # indexes of the elements to delete from x axis
        indexes = (
            list(range(0, crop_width + 1))
            + list(range(image.shape[2] - crop_width, image.shape[2]))
        )
    elif axis == 'y':
        # indexes of the elements to delete from y axis
        indexes = (
            list(range(0, crop_width + 1))
            + list(range(image.shape[1] - crop_width, image.shape[1]))
        )
    else:
        raise ValueError('axis should be either x or y')
    
    return np.delete(image.copy(), indexes, axis=(2 if axis == 'x' else 1))


def _resize_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    res_image = image.copy()
    image_width = image.shape[2]
    image_height = image.shape[1]

    if width > image_width:
        res_image = _pad_image(res_image, (width - image_width) // 2, axis='x')
    elif width < image_width:
        res_image = _crop_image(res_image, (image_width - width) // 2, axis='x')
    
    if height > image_height:
        res_image = _pad_image(res_image, (height - image_height) // 2, axis='y')
    elif height < image_height:
        res_image = _crop_image(res_image, (image_height - height) // 2, axis='y')

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

