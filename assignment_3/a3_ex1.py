import os
import glob
import numpy as np

from typing import Optional
from torch.utils.data import Dataset
from PIL import Image
from a2_ex1 import to_grayscale
from a2_ex2 import prepare_image


class ImagesDataset(Dataset):
	def __init__(self, image_dir, width: int = 100, height: int = 100, dtype: Optional[type] = None) -> None:
		super().__init__()
		if width < 100 or height < 100:
			raise ValueError('Width and height should be >= 100')
		
		self._image_paths = []
		for image_path in glob.glob(f'{image_dir}/*.jpg'):
			self._image_paths.append(os.path.abspath(image_path))
		self._image_paths.sort()
		
		# take the first found csv file in the image dir
		csv_file = glob.glob(f'{image_dir}/*.csv')[0]
		
		# convert csv file into np.ndarray skipping column names 
		self._labels = np.genfromtxt(csv_file, delimiter=';', skip_header=1, dtype=str)
		
		# sort by class name in ascending order
		self._labels.sort(axis=1)

		self._width = width
		self._height = height
		self._dtype = dtype
	
	def __getitem__(self, index: int) -> tuple:
		image_path = self._image_paths[index]
		image = Image.open(image_path)
		image_ndarray = np.asarray(image, dtype=self._dtype)
		image_ndarray = to_grayscale(image_ndarray)
		image_ndarray, subarea = prepare_image(image_ndarray, self._width, self._height, 0, 0, 32)

		class_name = self._labels[index][1]

		# class id corresponds to index
		# (image, class_id, class_name, image_filepath)
		return (image_ndarray, index, class_name, image_path)
	
	def __len__(self) -> int:
		return len(self._image_paths)

