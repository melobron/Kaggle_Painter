from torch.utils.data import Dataset

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


class Photo2Monet(Dataset):
    def __init__(self, root, transform=None):
        super(Photo2Monet, self).__init__()

        img_paths = sorted(make_dataset(root))
        if len(img_paths) == 0:
            raise RuntimeError('Found 0 images in: {}'.format(root))

        self.root = root
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img_numpy = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(img_numpy)
        return img_tensor

    def __len__(self):
        return len(self.img_paths)




