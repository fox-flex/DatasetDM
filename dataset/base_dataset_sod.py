import random
import importlib
import albumentations as A
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2


def get_dataset(dataset_name, **kwargs):
    dataset_name = dataset_name.lower()
    dataset_lib = importlib.import_module(
        '.' + dataset_name, package='dataset')

    dataset_abs = getattr(dataset_lib, dataset_name)
    print(dataset_abs)

    return dataset_abs(**kwargs)


class BaseDataset(Dataset):
    def __init__(self, crop_size):
        
        self.count = 0
        size = 512
        
        basic_transform = [
            A.LongestMaxSize(size, cv2.INTER_AREA),
            A.PadIfNeeded(
                min_height=size,
                min_width=size,
                border_mode=cv2.BORDER_CONSTANT,
                value=128,
            ),
            # A.HorizontalFlip(),
            # A.RandomCrop(crop_size[0], crop_size[1]),
            # A.RandomBrightnessContrast(),
            # A.RandomGamma(),
            # A.HueSaturationValue()
        ]
        self.basic_transform = basic_transform    
        self.to_tensor = transforms.ToTensor()

    def readTXT(self, txt_path):
        with open(txt_path, 'r') as f:
            listInTXT = [line.strip() for line in f]

        return listInTXT

    def augment_training_data(self, image, mask):
        H, W, C = image.shape

        additional_targets = {'mask': 'mask'}
        aug = A.Compose(transforms=self.basic_transform,
                        additional_targets=additional_targets)
        augmented = aug(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']

        image = self.to_tensor(image)
        mask = self.to_tensor(mask).squeeze()

        self.count += 1

        return image, mask

    def augment_test_data(self, image, mask):
        image = self.to_tensor(image)
        mask = self.to_tensor(mask).squeeze()

        return image, mask

