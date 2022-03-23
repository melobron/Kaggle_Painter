import os
import cv2


def is_image_file(filename):
    extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']
    return any(filename.endswith(extension) for extension in extensions)


def make_dataset(dir):
    img_paths = []
    assert os.path.isdir(dir), '{} is not a valid directory'.format(dir)

    for (root, dirs, files) in sorted(os.walk(dir)):
        for filename in files:
            if is_image_file(filename):
                img_paths.append(os.path.join(root, filename))
    return img_paths