import numpy as np


def to_grayscale(pil_image: np.ndarray) -> np.ndarray:
    if pil_image.ndim == 2:
        return pil_image.reshape((1, pil_image.shape[0], pil_image.shape[1]))
    if pil_image.ndim != 3:
        raise ValueError('Image does not have a (H, W) or (H, W, 3) shape')
    if pil_image.shape[2] != 3:
        raise ValueError('Image\'s 3rd dimension does not have a size 3')

    norm_image = pil_image / 255

    rgb_linear = np.where(norm_image <= 0.04045,
                          norm_image / 12.92,
                          ((norm_image + 0.055) / 1.055)**2.4)

    r_linear = np.delete(rgb_linear, [1, 2], axis=2)
    g_linear = np.delete(rgb_linear, [0, 2], axis=2)
    b_linear = np.delete(rgb_linear, [0, 1], axis=2)

    y_linear = 0.2126 * r_linear + 0.7152 * g_linear + 0.0722 * b_linear

    y = np.where(y_linear <= 0.0031308,
                 12.92 * y_linear,
                 1.055 * (y_linear ** (1/2.4)) - 0.055)

    y_reshaped = y.reshape((y.shape[2], y.shape[0], y.shape[1]))

    if np.issubdtype(pil_image.dtype, np.integer):
        return y_reshaped.round(0).astype(pil_image.dtype)
    return y_reshaped.astype(pil_image.dtype)

