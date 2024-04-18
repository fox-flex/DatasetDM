# ------------------------------------------------------------------------------
# The code is from GLPDepth (https://github.com/vinvino02/GLPDepth).
# For non-commercial purpose only (research, evaluation etc).
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Zhejiang University.
# author: weijia wu
# ------------------------------------------------------------------------------


import os
import cv2
import random
from dataset.base_dataset_sod import BaseDataset
import json
from tqdm.auto import tqdm


class UHRSD(BaseDataset):
    def __init__(
            self,
            data_path,
            is_train=True,
            image_limitation =50,
            crop_size=(512, 512)
    ):
        super().__init__(crop_size)
        
        self.is_train = is_train
        self.size=512
        self.image_limitation = image_limitation
        self.data_root = data_path
        
        # join paths if data_root is specified  data/kitti/kitti_eigen_train.txt
        self.img_dir = os.path.join(self.data_root, "image")
        self.mask_dir = os.path.join(self.data_root, "mask")
        self.cap_dir = os.path.join(self.data_root, "caption")
        self.cache_dir = os.path.join(self.data_root, 'cache')
        
        # load annotations
        self.img_infos = self.load_annotations(self.img_dir, self.mask_dir, self.cap_dir)
        random.shuffle(self.img_infos)
        self.img_infos = self.img_infos[:self.image_limitation]

    def load_annotations(self, img_dir, mask_dir, cap_dir):
        """Load annotation from directory.
        Args:
            img_dir (str): Path to image directory
            ann_dir (str|None): Path to annotation directory.
            cap_dir (str|None): Split txt file. Split should be specified, only file in the splits will be loaded.
        Returns:
            list[dict]: All image info of dataset.
        """
        img_infos = []

        for img_name in tqdm(os.listdir(self.img_dir), desc='Loading UHRSD dataset'):
            name = '.'.join(img_name.split('.')[:-1])
            mask_name = f'{name}.png'
            cap_name = f'{name}.txt'
            cache_name = f'{name}.pkl'
            assert os.path.exists(os.path.join(self.mask_dir, mask_name)), \
                f"mask {os.path.join(self.mask_dir, mask_name)} not found"
            assert os.path.exists(os.path.join(self.cap_dir, cap_name)), \
                f"caption {os.path.join(self.cap_dir, cap_name)} not found"
            
            img_infos.append((img_name, mask_name, cap_name, cache_name))

        img_infos = sorted(img_infos, key=lambda x: x[0])
        return img_infos
    
    def __len__(self):
        return len(self.img_infos)
    
    def __getitem__(self, idx):
        img_name, mask_name, cap_name, cache_name = self.img_infos[idx]

        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        cap_path = os.path.join(self.cap_dir, cap_name)
        cache_path = os.path.join(self.cache_dir, cache_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype('uint8') * 255
        assert mask.shape == image.shape[:2], f"mask shape {mask.shape} not equal to image shape {img.shape}"

        with open(cap_path, 'r') as f:
            cap = f.read().strip()
        assert cap != "", f"caption of {cap_name} is empty"

        original_mask = mask.copy()
        original_image = image.copy()
        
        
        if self.is_train:
            image, mask = self.augment_training_data(image, mask)
        else:
            image, mask = self.augment_test_data(image, mask)
        
        if False and os.path.exists(cache_path):
            with open(cache_path, 'rb') as cache_in:
                cache = pickle.load(cache_in)
        else:
            cache = cache_path
        
        return {
            'image': image,
            'mask': mask,
            'filename': img_path,
            'original_image':original_image,
            'original_mask': original_mask,
            "prompt": f"a photo of {cap}",
            "cache": cache,
        }
