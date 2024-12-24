# src/train_birads.py
import argparse
import yaml
import os
from pathlib import Path
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report, confusion_matrix, f1_score
from data.dataset import MammographyDataset, print_class_distribution
from utils.visualization import plot_confusion_matrix
from utils.metrics import calculate_metrics, calculate_class_weights
from models.birads_classifier import BiradsClassifier
from utils.metrics import calculate_metrics
import seaborn as sns
import matplotlib.pyplot as plt
import csv
from datetime import datetime
import pytz

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
    sydney_tz = pytz.timezone('Australia/Sydney')
    timestamp = datetime.now(sydney_tz).strftime('%Y%m%d_%H%M%S')
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
    train_dataset = MammographyDataset(metadata_path, split='train', task='birads', config=config)
    test_dataset = MammographyDataset(metadata_path, split='test', task='birads', config=config)
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    # Print class distribution and get class names
    class_names = print_class_distribution(train_dataset, 'birads')
    print_class_distribution(test_dataset, 'birads')
    
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
    
    # Calculate class weights
    class_weights = calculate_class_weights(train_dataset)
    print("\nClass weights:", class_weights)
    
    # Update loss function with class weights
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), 
                               lr=config['model']['birads_classifier']['learning_rate'])
    
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
            f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
            
            # Log metrics
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, 
                               running_loss/len(train_loader),
                               metrics['accuracy'],
                               f1,
                               metrics['kappa']])
            
            # Save best model with new naming format
            if metrics['accuracy'] > best_accuracy:
                best_accuracy = metrics['accuracy']
                model_save_name = f"{model_name}_acc{metrics['accuracy']:.3f}_epoch{epoch+1}.pth"
                best_model_path = os.path.join(model_dir, model_save_name)
                
                # Save confusion matrix
                cm_save_path = os.path.join(log_dir, f'confusion_matrix_epoch{epoch+1}.png')
                plot_confusion_matrix(all_labels, all_preds, 
                                    classes=class_names,
                                    save_path=cm_save_path)
                
                # Save model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': best_accuracy,
                    'f1_score': f1,
                    'model_name': model_name,
                }, best_model_path)
                
                print(f'\nNew best model saved: {model_save_name}')
        
            print(f'Epoch [{epoch+1}/{config["model"]["birads_classifier"]["epochs"]}]')
            print(f'Accuracy: {metrics["accuracy"]:.4f}, Kappa Score: {metrics["kappa"]:.4f}')
            print(f'F1-Score: {f1:.4f}')
            print('\nClassification Report:')
            print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

    print(f"\nTraining completed. Best model saved at: {best_model_path}")
    print(f"Training logs saved at: {log_file}")

    # Verify log file
    print("\nVerifying log file contents:")
    with open(log_file, 'r') as f:
        log_contents = f.read()
        print(log_contents)

if __name__ == '__main__':
    args = parse_args()
    train_birads(config_path=args.config, model_name=args.model, resume_path=args.resume)