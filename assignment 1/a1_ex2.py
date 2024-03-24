import os
import shutil
import re

from PIL import Image


_valid_extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG']
_valid_image_modes = ['RGB', 'L']
_max_file_size = 250_000


def _scan_dir_0(dir: str, file_paths: list[str]) -> None:
    for file in os.scandir(dir):
        if file.is_file():
            file_paths.append(os.path.abspath(file.path))
        else:
            _scan_dir_0(file.path, file_paths)


def _scan_dir(dir: str) -> list[str]:
    """
    Scans {dir} directory recursively and searches for files.

    :param dir: Directory where files are recursively scanned
    :raises ValueError: If dir is not an existing directory
    :return: A list of strings representing files' absolute paths
    """
    file_paths = []
    if not os.path.isdir(dir):
        raise ValueError(f'{dir} is not an existing directory')
    _scan_dir_0(dir, file_paths)
    return file_paths


def _is_image_variance_valid(image: Image.Image) -> bool:
    r_channel = list(image.getdata(0))
    r_channel_var = (max(r_channel) - min(r_channel))
    b_channel = list(image.getdata(1))
    b_channel_var = (max(b_channel) - min(b_channel))
    g_channel = list(image.getdata(2))
    g_channel_var = (max(g_channel) - min(g_channel))
    return not (r_channel_var == b_channel_var == g_channel_var == 0)


def validate_images(input_dir: str, output_dir: str,
                    log_file: str, formatter: str = '07d') -> None:
    """
        Validates images, copies valid images into the {output_dir} directory and
        then gives the copied images names based on {formatter}. It also writes
        logs into {log_file} about invalid images and creates "labels.csv" file
        in {output_dir} directory with image files' names and the corresponding
        labels. 

        Validation bases on the following rules:
                1. The file name ends with .jpg, .JPG, .jpeg or .JPEG.
                2. The file size does not exceed 250kB (=250 000 Bytes).
                3. The file can be read as image (i.e., the PIL/pillow module does not raise an exception
                when reading the file).
                4. The image data has a shape of (H, W, 3) with H (height) and W (width) larger than or
                equal to 100 pixels, and the three channels must be in the order RGB (red, green, blue).
                Alternatively, the image can also be grayscale and have a shape of only (H, W) with the
                same width and height restrictions.
                5. The image data has a variance larger than 0, i.e., there is not just one common pixel in
                the image data.
                6. The same image has not been copied already.

    :param input_dir: Input directory where images are looked for recursively.
    :param output_dir: Output directory where all valide images are copied to
                and labels.csv file is created. If the specified file or intermediate
                directories do not exist, they are created automatically.
    :param log_file: Path to the log file. If the specified file or
        intermediate directories do not exist, they will be created
        automatically. In case file exist, it will be truncated.
    :param formatter: optional format string used when writing the base names
                of the output validated images. For '07d', images' names will look
                like this: "000000{i}.jpg", where {i} is some index.
    :raises ValueError:
                - If input_dir is a path to a directory that
                does not exist, or;
                - If log_file is a path to an existing directory
                - If labels.csv is an existing directory inside output_dir
    """
    file_paths = sorted(_scan_dir(input_dir))

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.split(log_file)[0], exist_ok=True)

    labels_file = os.path.join(output_dir, 'labels.csv')
    # add columns to the csv file
    with open(labels_file, 'w') as labels_f:
        labels_f.write('name;label\n')

    image_hashes = set()

    with open(log_file, 'w') as log_f, open(labels_file, 'a') as labels_f:
        for index, file_path in enumerate(file_paths):
            file_name = os.path.split(file_path)[1]
            file_ext = os.path.splitext(file_path)[1]

            if file_ext not in _valid_extensions:
                log_f.write(f'{file_name},1\n')
                continue

            if os.path.getsize(file_path) > _max_file_size:
                log_f.write(f'{file_name},2\n')
                continue

            try:
                image = Image.open(file_path)
            except:
                log_f.write(f'{file_name},3\n')
                continue

            if image.size[0] < 100 or image.size[1] < 100 \
                    or image.mode not in _valid_image_modes:
                log_f.write(f'{file_name},4\n')
                continue

            if not _is_image_variance_valid(image):
                log_f.write(f'{file_name},5\n')
                continue
			
            image_hash = hash(tuple(image.getdata()))
            if image_hash in image_hashes:
                log_f.write(f'{file_name},6\n')
                continue

            formatted_name = ('{:' + formatter + '}').format(index) + '.jpg'
            # delete all integers in a base file name without an extension
            label = re.sub(r'\d+', '', os.path.splitext(file_name)[0])

            labels_f.write(f'{formatted_name};{label}\n')
            shutil.copy(file_path, os.path.join(output_dir, formatted_name))

            image_hashes.add(image_hash)

validate_images('./exifless', './output', './log_file.log')
