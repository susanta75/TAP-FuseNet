import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob
import cv2
import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2


class NormalDataset(Dataset):
    def __init__(self, data_path, transform, mode='train'):
        self.imgs_list = sorted(glob(data_path + "/image/*"))
        self.masks_list = sorted(glob(data_path + "/mask/*"))
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.imgs_list)

    def __getitem__(self, index):
        img_path = self.imgs_list[index]
        mask_path = self.masks_list[index]
        mask_name = mask_path.split("/")[-1]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.mode != 'train':
            ori_mask = torch.from_numpy(mask).float() / 255.0

        augmented = self.transform(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask'].float() / 255.0

        if self.mode == 'train':
            return {'img': img, 'mask': mask}
        else:
            return {'img': img, 'mask': mask, 'ori_mask': ori_mask, 'mask_name': mask_name}


def get_augmentation(version=0, img_size=512):
    if version == 0:  # Training augmentation
        transforms = albu.Compose([
            albu.OneOf([
                albu.HorizontalFlip(),
                albu.VerticalFlip(),
                albu.RandomRotate90()
            ], p=0.5),
            albu.OneOf([
                albu.MotionBlur(blur_limit=5),
                albu.MedianBlur(blur_limit=5),
                albu.GaussianBlur(blur_limit=5),
                albu.GaussNoise(var_limit=(5.0, 20.0)),  # warning will appear but ignored
            ], p=0.5),
            albu.Resize(img_size, img_size),
            albu.Normalize([0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:  # Testing augmentation
        transforms = albu.Compose([
            albu.Resize(img_size, img_size),
            albu.Normalize([0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    return transforms


def getSODDataloader(data_path, batch_size, num_workers, mode, img_size=512):
    if mode == "train":
        transform = get_augmentation(0, img_size)
    else:
        transform = get_augmentation(1, img_size)

    dataset = NormalDataset(data_path + "/" + mode, transform, mode)

    dataLoader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == "train"),
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataLoader

