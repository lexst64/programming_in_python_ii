from PIL import Image
from assignment_2.a2_ex1 import to_grayscale
from assignment_2.a2_ex2 import prepare_image
import numpy as np


im = Image.open('./assignment_1/output/0000029.jpg')
gs_arr = to_grayscale(np.array(im).astype(np.float64))
Image.fromarray(gs_arr[0]).save('./test_gs.tiff')

init_width = 767
init_height = 1019

resize, subarea = prepare_image(gs_arr, init_width, init_height - 300, 200, 200, 100)
Image.fromarray(resize[0]).save('./test_resize.tiff')
Image.fromarray(subarea[0]).save('./test_subarea.tiff')
