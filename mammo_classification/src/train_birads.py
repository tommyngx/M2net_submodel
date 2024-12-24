# src/train_birads.py
import argparse
import yaml
import os
from pathlib import Path
import torch
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report
from data.dataset import MammographyDataset
from models.birads_classifier import BiradsClassifier
from utils.metrics import calculate_metrics
import csv
from datetime import datetime

def get_supported_models():
    return ['resnet50', 'efficientnet', 'vit', 'convnext']

def parse_args():
    parser = argparse.ArgumentParser(description='Train BIRADS classifier')
    parser.add_argument('--config', type=str, default='config/model_config.yaml',
                      help='path to config file')
    parser.add_argument('--model', type=str, default='resnet50',
                      choices=get_supported_models(),
                      help='model architecture to use')
    parser.add_argument('--resume', type=str, default=None,
                      help='path to previous model checkpoint')
    return parser.parse_args()

def train_birads(config_path='config/model_config.yaml', model_name='resnet50', resume_path=None):
    # Load config first
    print("1. Loading configuration...")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Create output directories from config
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = config['output']['base_dir']
    output_dir = os.path.join(base_dir, 'birads', timestamp)
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
    metrics_columns = ['epoch', 'train_loss'] + config['output']['metrics']
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
    train_dataset = MammographyDataset(metadata_path, split='train', task='birads')
    test_dataset = MammographyDataset(metadata_path, split='test', task='birads')
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, 
                            batch_size=config['model']['birads_classifier']['batch_size'],
                            shuffle=True)
    test_loader = DataLoader(test_dataset, 
                           batch_size=config['model']['birads_classifier']['batch_size'])
    
    # Initialize model with config parameters
    print("\n3. Initializing model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = BiradsClassifier(
        model_name=model_name,
        num_classes=train_dataset.num_classes,
        pretrained=config['model']['birads_classifier']['pretrained']
    ).to(device)
    
    # Load previous checkpoint if specified
    start_epoch = 0
    if resume_path:
        if os.path.isfile(resume_path):
            print(f"Loading checkpoint: {resume_path}")
            checkpoint = torch.load(resume_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_accuracy = checkpoint['accuracy']
            print(f"Resumed from epoch {start_epoch} with accuracy: {best_accuracy:.4f}")
        else:
            print(f"No checkpoint found at: {resume_path}")
    
    # Training setup
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 
                               lr=config['model']['birads_classifier']['learning_rate'])
    
    print("\n4. Starting training...")
    # Training loop
    for epoch in range(start_epoch, config['model']['birads_classifier']['epochs']):
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["model"]["birads_classifier"]["epochs"]}')
        for i, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{running_loss/(i+1):.4f}'})
            
        # Evaluation
        if (epoch + 1) % 2 == 0:
            model.eval()
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.numpy())
            
            # Calculate metrics
            metrics = calculate_metrics(all_labels, all_preds)
            
            # Log metrics
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, 
                               running_loss/len(train_loader),
                               metrics['accuracy'],
                               metrics['kappa']])
            
            # Save best model with new naming format
            if metrics['accuracy'] > best_accuracy:
                best_accuracy = metrics['accuracy']
                model_save_name = f"{model_name}_acc{metrics['accuracy']:.3f}_epoch{epoch+1}.pth"
                best_model_path = os.path.join(model_dir, model_save_name)
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': best_accuracy,
                    'model_name': model_name,
                }, best_model_path)
                
                print(f'\nNew best model saved: {model_save_name}')
            
            print(f'\nEpoch [{epoch+1}/{config["model"]["birads_classifier"]["epochs"]}]')
            print(f'Average Loss: {running_loss/len(train_loader):.4f}')
            print(f'Validation Accuracy: {metrics["accuracy"]:.4f}')
            print(f'Validation Kappa: {metrics["kappa"]:.4f}')

    print(f"\nTraining completed. Best model saved at: {best_model_path}")
    print(f"Training logs saved at: {log_file}")

if __name__ == '__main__':
    args = parse_args()
    train_birads(config_path=args.config, model_name=args.model, resume_path=args.resume)