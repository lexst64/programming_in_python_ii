import torch
import numpy as np


def stacking(batch_as_list: list) -> tuple:
	stacked_images = torch.from_numpy(np.array([image for image, _, _, _ in batch_as_list]))
	stacked_class_ids = torch.from_numpy(np.array([class_id for _, class_id, _, _ in batch_as_list]))	
	class_names = [class_name for _, _, class_name, _ in batch_as_list]
	image_filepaths = [image_filepath for _, _, _, image_filepath in batch_as_list]

	return stacked_images, stacked_class_ids, class_names, image_filepaths


