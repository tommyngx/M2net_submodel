# src/train_birads.py
import yaml
import os
from pathlib import Path
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report
from data.dataset import MammographyDataset
from models.birads_classifier import BiradsClassifier
from utils.metrics import calculate_metrics

def train_birads(config_path='config/model_config.yaml'):
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Get metadata path from config
    metadata_path = config['data']['metadata_path']
    
    # Validate path
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at: {metadata_path}")
        
    # Create datasets
    train_dataset = MammographyDataset(metadata_path, split='train', task='birads')
    test_dataset = MammographyDataset(metadata_path, split='test', task='birads')
    
    train_loader = DataLoader(train_dataset, 
                            batch_size=config['model']['birads_classifier']['batch_size'],
                            shuffle=True)
    test_loader = DataLoader(test_dataset, 
                           batch_size=config['model']['birads_classifier']['batch_size'])
    
    # Initialize model with config parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BiradsClassifier(
        model_name=config['model']['birads_classifier']['model_name'],
        num_classes=train_dataset.num_classes,
        pretrained=config['model']['birads_classifier']['pretrained']
    ).to(device)
    
    # Training setup
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 
                               lr=config['model']['birads_classifier']['learning_rate'])
    
    # Training loop
    for epoch in range(config['model']['birads_classifier']['epochs']):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        # Evaluation
        if (epoch + 1) % 5 == 0:
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
            
            # Calculate all metrics at once
            metrics = calculate_metrics(all_labels, all_preds)
            
            print(f'Epoch [{epoch+1}/{config["model"]["birads_classifier"]["epochs"]}]')
            print(f'Accuracy: {metrics["accuracy"]:.4f}, Kappa Score: {metrics["kappa"]:.4f}')
            print(f'Precision: {metrics["precision"]:.4f}, Recall: {metrics["recall"]:.4f}, F1: {metrics["f1"]:.4f}')

if __name__ == '__main__':
    train_birads()