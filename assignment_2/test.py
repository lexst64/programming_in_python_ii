from PIL import Image
from a2_ex1 import to_grayscale
from a2_ex2 import prepare_image
import numpy as np


im = Image.open('./assignment_1/output/0000029.jpg')
gs_arr = to_grayscale(np.array(im).astype(np.float64))
# gs_arr = np.pad(gs_arr, ((0, 0), (1, 0), (0, 0)))
Image.fromarray(gs_arr[0]).save('./test_gs.tiff')

init_width = gs_arr.shape[2]
init_height = gs_arr.shape[1]

resize, subarea = prepare_image(gs_arr, 1280, 720, 200, 200, 300)
Image.fromarray(resize[0]).save('./test_resize.tiff')
Image.fromarray(subarea[0]).save('./test_subarea.tiff')
