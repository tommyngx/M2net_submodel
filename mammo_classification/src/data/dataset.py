import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class MammographyDataset(Dataset):
    def __init__(self, metadata_path, split, task, config):
        self.df = pd.read_csv(metadata_path)

        old_prefix = '/content/'
        new_prefix = config['data']['image_dir']

        self.df['path'] = self.df['path'].str.replace(old_prefix, new_prefix, regex=False)
        self.df['path2'] = self.df['path2'].str.replace(old_prefix, new_prefix, regex=False)

        upgrade_df_with_image_size_and_save(metadata_path, path_column='path2', size_threshold=20)
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
            A.Normalize(),
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


import pandas as pd
import os

def upgrade_df_with_image_size_and_save(csv_link, path_column='path2', size_column='size', size_threshold=10):
    """
    Updates the DataFrame by adding a column with image sizes (in KB), filters out rows with images below a size threshold,
    and saves the updated DataFrame back to the same CSV file.

    Parameters:
    - csv_link (str): Path to the CSV file containing the DataFrame.
    - path_column (str): The name of the column containing the image paths.
    - size_column (str): The name of the new column to store image sizes in KB.
    - size_threshold (int): The minimum file size (in KB) to retain an image.

    Returns:
    - None
    """
    def get_image_size(image_path):
        try:
            if os.path.exists(image_path):
                size_bytes = os.path.getsize(image_path)
                return size_bytes / 1024  # Convert bytes to KB
            else:
                return 0
        except Exception as e:
            print(f"Error reading image size for {image_path}: {e}")
            return 0

    # Load the DataFrame from the CSV file
    df = pd.read_csv(csv_link)

    # Add the size column by calculating the size for each image
    df[size_column] = df[path_column].apply(get_image_size)

    # Filter out rows where the image size is less than the threshold
    df = df[df[size_column] >= size_threshold].reset_index(drop=True)

    # Save the updated DataFrame back to the same CSV file
    df.to_csv(csv_link, index=False)
    print(f"Updated DataFrame saved to {csv_link}")

# Example usage:
# 