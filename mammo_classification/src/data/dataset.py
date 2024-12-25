import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import os

class MammographyDataset(Dataset):
    def __init__(self, metadata_path, split, task, config):
        self.df = pd.read_csv(metadata_path)

        old_prefix = '/content/'
        new_prefix = config['data']['image_dir']

        self.df['path'] = self.df['path'].str.replace(old_prefix, new_prefix, regex=False)
        self.df['path2'] = self.df['path2'].str.replace(old_prefix, new_prefix, regex=False)

        upgrade_df_with_image_size_and_save(self.df, metadata_path, path_column='path2')
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
        resize_dims = augmentation_config['resize']
        self.transform = A.Compose([
            A.HorizontalFlip(p=augmentation_config['horizontal_flip']),
            A.VerticalFlip(p=augmentation_config['vertical_flip']),
            A.Rotate(limit=augmentation_config['rotation_range']),
            A.RandomScale(scale_limit=augmentation_config['zoom_range']),
            A.RandomBrightnessContrast(brightness_limit=augmentation_config['brightness_range']),
            A.Resize(resize_dims[0], resize_dims[1]),
            #A.Normalize(),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['path2']).convert('RGB')
        image = np.array(image)
        
        if self.transform:
            image = self.transform(image=image)['image']
            
        if self.task == 'birads':
            label = self.label_to_idx[row['birads']]
        else:
            label = self.label_to_idx[row['lesion_types']]
            
        return image, label

def print_class_distribution(dataset, task):
    """Print the number of images for each class and return class names"""
    class_counts = dataset.df[task].value_counts()
    class_names = class_counts.index.tolist()
    print(f"\nClass distribution for {task}:")
    for class_label, count in class_counts.items():
        print(f"Class {class_label}: {count} images")
    return class_names


def upgrade_df_with_image_size_and_save(df, metadata_path, path_column='path2', size_threshold=3):
    """Add image size columns and remove small files from dataframe"""
    # Read existing dataframe
    #df = pd.read_csv(metadata_path)
    initial_count = len(df)
    
    # Add size columns
    df['width'] = None
    df['height'] = None
    df['file_size_kb'] = None
    
    # Update size information
    for idx, row in df.iterrows():
        try:
            img_path = row[path_column]
            if os.path.exists(img_path):
                file_size = os.path.getsize(img_path) / 1024
                with Image.open(img_path) as img:
                    width, height = img.size
                df.at[idx, 'width'] = width
                df.at[idx, 'height'] = height 
                df.at[idx, 'file_size_kb'] = file_size
        except Exception as e:
            df.at[idx, 'file_size_kb'] = 0
            continue
    
    # Remove small files
    df = df[df['file_size_kb'] >= size_threshold]
    final_count = len(df)
    
    # Save updated dataframe
    df.to_csv(metadata_path, index=False)
    print(f"Files before: {initial_count}, after: {final_count} (removed {initial_count - final_count} files < {size_threshold}KB)")