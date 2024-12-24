# src/train_lesion.py
import argparse
import yaml
import torch
import os
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report
from data.dataset import MammographyDataset, print_class_distribution
from utils.visualization import plot_confusion_matrix
from utils.metrics import calculate_metrics, calculate_class_weights
from models.lesion_classifier import LesionClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
import csv
from datetime import datetime
import pytz

def get_supported_models():
    return ['resnet50', 'efficientnet', 'vit', 'convnext']

def parse_args():
    parser = argparse.ArgumentParser(description='Train Lesion classifier')
    parser.add_argument('--config', type=str, default='config/model_config.yaml',
                      help='path to config file')
    parser.add_argument('--model', type=str, default='resnet50',
                      choices=get_supported_models(),
                      help='model architecture to use')
    parser.add_argument('--resume', type=str, default=None,
                      help='path to previous model checkpoint')
    return parser.parse_args()

def train_lesion(config_path='config/model_config.yaml', model_name='resnet50', resume_path=None):
    # Load config first
    print("1. Loading configuration...")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Create output directories from config
    sydney_tz = pytz.timezone('Australia/Sydney')
    timestamp = datetime.now(sydney_tz).strftime('%Y%m%d_%H%M%S')
    base_dir = config['output']['base_dir']
    output_dir = os.path.join(base_dir, 'lesion', timestamp)
    model_dir = os.path.join(output_dir, config['output']['model_dir'])
    log_dir = os.path.join(output_dir, config['output']['log_dir'])
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize logging paths
    log_file = os.path.join(log_dir, 'training_log.csv')
    best_model_path = os.path.join(model_dir, 'best_model.pth')
    
    # Initialize best metrics
    best_accuracy = 0.0
    
    # Create CSV logger with metrics from config
    metrics_columns = ['epoch', 'train_loss', 'accuracy', 'f1_score', 'kappa']
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(metrics_columns)
    
    # Get metadata path from config
    metadata_path = config['data']['metadata_path']
    
    # Validate path
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at: {metadata_path}")
        
    # Create datasets
    print("\n2. Creating datasets...")
    train_dataset = MammographyDataset(metadata_path, split='train', task='lesion_types', config=config)
    test_dataset = MammographyDataset(metadata_path, split='test', task='lesion_types', config=config)
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    # Print class distribution and get class names
    class_names = print_class_distribution(train_dataset, 'lesion_types')
    print_class_distribution(test_dataset, 'lesion_types')
    
    train_loader = DataLoader(train_dataset, 
                            batch_size=config['model']['lesion_classifier']['batch_size'],
                            shuffle=True)
    test_loader = DataLoader(test_dataset, 
                           batch_size=config['model']['lesion_classifier']['batch_size'])
    
    # Initialize model with config parameters
    print("\n3. Initializing model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = LesionClassifier(
        model_name=model_name,
        num_classes=train_dataset.num_classes,
        pretrained=config['model']['lesion_classifier']['pretrained']
    ).to(device)
    
    # Calculate class weights
    class_weights = calculate_class_weights(train_dataset)
    print("\nClass weights:", class_weights)
    
    # Update loss function with class weights
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), 
                               lr=config['model']['lesion_classifier']['learning_rate'])
    
    # Load previous checkpoint if specified
    start_epoch = 0
    if resume_path:
        if os.path.isfile(resume_path):
            print(f"Loading checkpoint: {resume_path}")
            checkpoint = torch.load(resume_path, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_accuracy = checkpoint['accuracy']
            print(f"Resumed from epoch {start_epoch} with accuracy: {best_accuracy:.4f}")
        else:
            print(f"No checkpoint found at: {resume_path}")

    # Rest of the training loop code follows the same pattern as train_birads.py
    ...

if __name__ == '__main__':
    args = parse_args()
    train_lesion(config_path=args.config, model_name=args.model, resume_path=args.resume)