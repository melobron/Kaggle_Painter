import torchvision.transforms as transforms

import os
import time
import yaml


################################# Modalities Encoder & Extractor #################################
def get_modalities_extraction_loader(config):
    transform = transforms.Compose([transforms.Resize(config['data']['new_size']),
                                    transforms.CenterCrop((config['data']['crop_image_height'], config['data']['crop_image_width'])),
                                    transforms.ToTensor()])


################################# Path & Directory #################################
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


def get_config(config):
    with open(config, 'r') as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def make_result_folders(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        os.makedirs(image_directory)

    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        os.makedirs(checkpoint_directory)

    logs_directory = os.path.join(output_directory, "logs")
    if not os.path.exists(logs_directory):
        os.makedirs(logs_directory)
    return checkpoint_directory, image_directory, logs_directory


################################# Record #################################
def write_loss(iterations, trainer, train_writer):
    members = [attr for attr in dir(trainer)
               if ((not callable(getattr(trainer, attr))
                    and not attr.startswith("__"))
                   and ('loss' in attr
                        or 'grad' in attr
                        or 'nwd' in attr
                        or 'accuracy' in attr))]
    for m in members:
        tag = m.split('_')[1]
        train_writer.add_scalar(f"{tag}/{m}", getattr(trainer, m), iterations)


################################# ETC #################################
class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(self.msg.format(time.time() - self.start_time))




