import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class MammographyDataset(Dataset):
    def __init__(self, metadata_path, split, task, config):
        """
        Args:
            metadata_path: Path to CSV file with columns [path, birads, lesion_types, split]
            split: 'train' or 'test'
            task: 'birads' or 'lesion'
            config: Configuration dictionary containing augmentation parameters
        """
        self.df = pd.read_csv(metadata_path)
        self.df = self.df[self.df['split'] == split]
        self.task = task
        
        # Get number of classes
        if task == 'birads':
            self.labels = self.df['birads'].unique()
        else:
            self.labels = self.df['lesion_types'].unique()
        
        self.num_classes = len(self.labels)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        
        # Load augmentations from config
        augmentation_config = config['data']['augmentation']
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=augmentation_config['horizontal_flip']),
            A.VerticalFlip(p=augmentation_config['vertical_flip']),
            A.Rotate(limit=augmentation_config['rotation_range']),
            A.RandomScale(scale_limit=augmentation_config['zoom_range']),
            A.RandomBrightnessContrast(brightness_limit=augmentation_config['brightness_range']),
            A.Normalize(),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['path']).convert('RGB')
        image = np.array(image)
        
        if self.transform:
            image = self.transform(image=image)['image']
            
        if self.task == 'birads':
            label = self.label_to_idx[row['birads']]
        else:
            label = self.label_to_idx[row['lesion_types']]
            
        return image, label