import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from utils import *


class ModalityDataset(Dataset):
    def __init__(self, root, transform=None):
        super(ModalityDataset, self).__init__()

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


class EncoderDataset:
    def __init__(self, config):
        self.batch_size = config['modalities_encoder']['batch_size']
        self.color_jitter_strength = config['modalities_encoder']['color_jitter_strength']
        self.input_shape = (config['data']['crop_image_height'], config['data']['crop_image_width'], 3)
        self.augmentations = config['modalities_encoder']['augmentations']
        self.root = config['data']['train_root']

    def get_dataloader(self):
        data_augment = self.get_augmentations_transform()
        transform = AugDataTransform(data_augment)
        train_dataset = ModalityDataset(self.root, transform)
        sampler = SubsetRandomSampler(range(len(train_dataset)))
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=sampler, shuffle=False)
        return train_loader

    def get_augmentations_transform(self):
        augmentations = []
        if 'crop' in self.augmentations:
            augmentations.append(transforms.RandomResizedCrop(size=self.input_shape[0]))
        if 'horizontal_flip' in self.augmentations:
            augmentations.append(transforms.RandomHorizontalFlip())
        if 'color_jitter' in self.augmentations:
            color_jitter = transforms.ColorJitter(0.8 * self.color_jitter_strength, 0.8 * self.color_jitter_strength,
                                                  0.8 * self.color_jitter_strength, 0.2 * self.color_jitter_strength)
            augmentations.append(transforms.RandomApply([], p=0.8))
        if 'gray_scale' in self.augmentations:
            augmentations.append(transforms.RandomGrayscale(p=0.2))
        if 'blur' in self.augmentations:
            augmentations.append(transforms.GaussianBlur(kernel_size=int(0.1*self.input_shape[0]-1)))
        augmentations.append(transforms.ToTensor())
        transform = transforms.Compose(augmentations)
        return transform


class AugDataTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        xi = self.transform(sample)
        xj = self.transform(sample)
        return xi, xj
