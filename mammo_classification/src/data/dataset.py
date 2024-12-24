import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class MammographyDataset(Dataset):
    def __init__(self, metadata_path, split='train', task='birads', transform=None):
        """
        Args:
            metadata_path: Path to CSV file with columns [path, birads, lesion_types, split]
            split: 'train' or 'test'
            task: 'birads' or 'lesion'
            transform: Optional transform to be applied
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
        
        if transform is None:
            self.transform = A.Compose([
                A.Resize(224, 224),
                A.Normalize(),
                ToTensorV2()
            ])
        else:
            self.transform = transform

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