import numpy as np


def to_grayscale(pil_image: np.ndarray) -> np.ndarray:
    """
    Converts ``pil_image`` to grayscale using the colorimetric conversion.

    :param pil_image: the raw data of an image loaded with Pillow (PIL)
    :return: grayscaled image as ``np.ndarray`` shaped as (1, H, W), where W is
        the width and H is the height of the image.
    """
    if pil_image.ndim == 2:
        return pil_image.reshape((1, pil_image.shape[0], pil_image.shape[1]))
    if pil_image.ndim != 3:
        raise ValueError('Image does not have a (H, W) or (H, W, 3) shape')
    if pil_image.shape[2] != 3:
        raise ValueError('Image\'s 3rd dimension does not have a size 3')

    # we assume pil_image has values in [0, 255] and normalize them to [0, 1]
    rgb_norm = pil_image / 255

    # apply gamma expansion
    rgb_linear = np.where(rgb_norm <= 0.04045,
                          rgb_norm / 12.92,
                          ((rgb_norm + 0.055) / 1.055)**2.4)

    # leave linear R, G and B channels separate in (H, W, 1) shaped ndarrays
    r_linear = np.delete(rgb_linear, [1, 2], axis=2)
    g_linear = np.delete(rgb_linear, [0, 2], axis=2)
    b_linear = np.delete(rgb_linear, [0, 1], axis=2)

    # calculate linear luminance
    y_linear = 0.2126 * r_linear + 0.7152 * g_linear + 0.0722 * b_linear

    # get linear luminance back to a non-linear representation
    # by the inverse of the gamma expansion
    y = np.where(y_linear <= 0.0031308,
                 12.92 * y_linear,
                 1.055 * (y_linear ** (1/2.4)) - 0.055)

    # shape to (1, H, W)
    y_reshaped = y.reshape((1, y.shape[0], y.shape[1]))

    if np.issubdtype(pil_image.dtype, np.integer):
        return y_reshaped.round(0).astype(pil_image.dtype)
    return y_reshaped.astype(pil_image.dtype)

