import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import timm
import pandas as pd
import logging
import random
from albumentations.pytorch import ToTensorV2
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, PiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout,
    ShiftScaleRotate, CenterCrop, Resize, SmallestMaxSize
)
from tqdm import tqdm
import cv2


CFG = {
    'seed': 42,  # 719,42,68

    'model_arch': 'vit_small_patch16_224',
    'patch': 16,
    
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
    # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    # mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    # OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
    # OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

    'img_size': 224,
    'class_num': 1784,

    'train_bs': 32,
    'valid_bs': 64,

    'num_workers': 1,
    'device': 'cuda',

    'weight_decay': 2e-5,
    'accum_iter': 1,    # suppoprt to do batch accumulation for backprop with effectively larger batch size
    'verbose_step': 1,  # the step of printing loss
}


train_meta_path = r"F:\datasets\FGVC10\fungi\FungiCLEF2023_train_metadata_PRODUCTION.csv"
val_meta_path = r"F:\datasets\FGVC10\fungi\FungiCLEF2023_val_metadata_PRODUCTION.csv"
test_meta_path = r"F:\datasets\FGVC10\fungi\FungiCLEF2023_public_test_metadata_PRODUCTION.csv"

train_root = r"F:\datasets\FGVC10\fungi\DF20_300"
val_test_root = r"F:\datasets\FGVC10\fungi\DF21_300"

train_meta = pd.read_csv(train_meta_path)
val_meta = pd.read_csv(val_meta_path)
test_meta = pd.read_csv(test_meta_path)

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler(f"logs/{CFG['model_arch']}_extract_features.log")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_img(path):
    im_bgr = cv2.imread(path)
    im_rgb = im_bgr[:, :, ::-1]
    return im_rgb

class FungiDataset(Dataset):
    def __init__(self, df, data_root,
                 transforms=None,
                 output_label=True,
                 one_hot_label=False
                 ):

        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.data_root = data_root

        self.output_label = output_label
        self.one_hot_label = one_hot_label

        if output_label == True:
            self.labels = self.df['class_id'].values

            if one_hot_label is True:
                self.labels = np.eye(self.df['class_id'].max() + 1)[self.labels]

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index: int):
        # get labels
        if self.output_label:
            target = self.labels[index]

        image_path = os.path.join(self.data_root, self.df.loc[index]['image_path'])
        # print(self.df.loc[index]["observationID"], image_path)
        img = get_img(image_path)

        if self.transforms:
            img = self.transforms(image=img)['image']

        if self.output_label:
            return img, target
        else:
            return img

def get_valid_transforms():
    return Compose([
        # SmallestMaxSize(CFG['img_size']),
        Resize(CFG['img_size'], CFG['img_size'],
               interpolation=cv2.INTER_CUBIC),
        # CenterCrop(CFG['img_size'], CFG['img_size']),
        Normalize(mean=CFG['mean'], std=CFG['std'],
                  max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.)

def extract_features(model, dataloader, device, save_path):
    model.eval()
    features = []
    with torch.no_grad():
        for idx, (images) in tqdm(enumerate(dataloader)):
            images = images.to(device)
            features.append(model.forward_features(images).cpu().numpy())
    features = np.concatenate(features, axis=0)
    np.save(save_path, features)
    logger.info(f"features shape: {features.shape}")
    logger.info(f"features saved to: {save_path}")


if __name__ == "__main__":
    seed_everything(CFG['seed'])

    model = timm.create_model(CFG['model_arch'], pretrained=True)
    device = torch.device('cpu')
    if CFG['device'] == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    model.to(device)
    
    # train_dataset = FungiDataset(train_meta, train_val_root, transforms=get_valid_transforms(), output_label=False)
    # train_loader = DataLoader(train_dataset, batch_size=CFG['valid_bs'], shuffle=False, num_workers=CFG['num_workers'])
    # extract_features(model, train_loader, device, f"features/{CFG['model_arch']}_train_features.npy")

    # val_dataset = FungiDataset(val_meta, val_test_root, transforms=get_valid_transforms(), output_label=False)
    # val_loader = DataLoader(val_dataset, batch_size=CFG['valid_bs'], shuffle=False, num_workers=CFG['num_workers'])
    # extract_features(model, val_loader, device, f"features/{CFG['model_arch']}_val_features.npy")

    test_dataset = FungiDataset(test_meta, val_test_root, transforms=get_valid_transforms(), output_label=False)
    test_loader = DataLoader(test_dataset, batch_size=CFG['valid_bs'], shuffle=False, num_workers=CFG['num_workers'])
    extract_features(model, test_loader, device, f"features/{CFG['model_arch']}_test_features.npy")
