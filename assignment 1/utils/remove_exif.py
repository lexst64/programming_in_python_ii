from PIL import Image

import os
import argparse
from tqdm import tqdm


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("input_dir",
						type=str,
						help="The directory containing the images.")
arg_parser.add_argument("--output_dir",
						type=str,
						help="The directory containing the resized images. If not specified, the original 'input_dir' with "
						"the additional postfix '_resized' will be used (directory will be created).")

args = arg_parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
supported_extensions = {".jpg", ".JPG", ".jpeg", ".JPEG"}
image_files = [f.path
			   for f in os.scandir(args.input_dir)
			   if f.is_file() and os.path.splitext(f)[1] in supported_extensions]

for image_file in tqdm(image_files):
	file_name = os.path.basename(image_file)
	with Image.open(image_file) as image:
		new_image = Image.new(image.mode, image.size)
		new_image.putdata(image.getdata())
		new_image.save(os.path.join(args.output_dir, file_name))
