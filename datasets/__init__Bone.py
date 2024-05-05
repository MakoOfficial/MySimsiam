import os
import pandas as pd
import cv2
from torch.utils.data import Dataset
from albumentations.augmentations.functional import normalize

from albumentations.augmentations.transforms import Lambda, Normalize, RandomBrightnessContrast
from albumentations.augmentations.geometric.transforms import ShiftScaleRotate, HorizontalFlip
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.augmentations.crops.transforms import RandomResizedCrop
from albumentations import Compose, Resize


import torchvision.transforms as transforms

RandomErasing = transforms.RandomErasing(scale=(0.02, 0.08), ratio=(0.5, 2), p=0.8)

train_norm_mean = (0.147, 0.147, 0.147)
train_norm_std = (0.275, 0.275, 0.275)


def randomErase(image, **kwargs):
    return RandomErasing(image)


def sample_normalize(image, **kwargs):
    image = image / 255
    # channel = image.shape[2]
    # mean, std = image.reshape((-1, channel)).mean(axis=0), image.reshape((-1, channel)).std(axis=0)
    # return (image - mean) / (std + 1e-3)
    return normalize(image, mean=train_norm_mean, std=train_norm_std)

transform_train = Compose([
    Resize(height=256, width=256),
    RandomResizedCrop(256, 256, (0.5, 1.0), p=0.5),
    HorizontalFlip(p=0.5),
    ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, border_mode=cv2.BORDER_CONSTANT, value=0.0,
                     p=0.8),
    RandomBrightnessContrast(p=0.8, contrast_limit=(-0.3, 0.2)),
    Lambda(image=sample_normalize),
    ToTensorV2(),
    # Lambda(image=randomErase)
])

transform_val = Compose([
    Lambda(image=sample_normalize),
    ToTensorV2(),
    # Normalize(mean=train_norm_mean, std=train_norm_std),
])


class BAADataset(Dataset):
    def __init__(self, df, file_path, isTrain):
        def preprocess_df(df):
            # nomalize boneage distribution
            # df['zscore'] = df['boneage'].map(lambda x: (x - boneage_mean) / boneage_div)
            # change the type of gender, change bool variable to float32
            df['bonage'] = df['boneage'].astype('float32')
            return df

        self.df = preprocess_df(df)
        self.file_path = file_path
        self.isTrain = isTrain

    def __getitem__(self, index):
        row = self.df.iloc[index]
        num = int(row['id'])
        img = cv2.imread(f"{self.file_path}/{num}.png", cv2.IMREAD_COLOR)

        if self.isTrain:
            transform = transform_train
        else:
            transform = transform_val
        return (transform(image=img)['image'], transform(image=img)['image']), row['boneage']

    def __len__(self):
        return len(self.df)


def get_dataset(data_dir, train=True):
    if train:
        filepath = os.path.join(data_dir, "train")
        df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    else:
        filepath = os.path.join(data_dir, "valid")
        df = pd.read_csv(os.path.join(data_dir, "valid.csv"))

    dataset = BAADataset(df=df, file_path=filepath, isTrain=train)
    return dataset