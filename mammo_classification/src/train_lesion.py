# src/train_lesion.py
import yaml
import torch
import os
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report
from data.dataset import MammographyDataset
from models.lesion_classifier import LesionClassifier
from utils.metrics import calculate_metrics
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score

def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_lesion(config_path='config/model_config.yaml', model_name='resnet50'):
    # Load config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
        
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Get metadata path from config
    metadata_path = config['data']['metadata_path']
    
    # Validate metadata path
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at: {metadata_path}")
    
    # Create datasets
    train_dataset = MammographyDataset(
        metadata_path=metadata_path, 
        split='train',
        task='lesion'
    )
    test_dataset = MammographyDataset(
        metadata_path=metadata_path, 
        split='test',
        task='lesion'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['model']['lesion_classifier']['batch_size'],
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['model']['lesion_classifier']['batch_size']
    )
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LesionClassifier(num_classes=train_dataset.num_classes).to(device)
    
    # Training setup
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), 
                               lr=config['model']['lesion_classifier']['learning_rate'])
    
    best_accuracy = 0.0
    model_dir = config['model']['lesion_classifier']['model_dir']
    log_dir = config['model']['lesion_classifier']['log_dir']
    
    # Training loop
    for epoch in range(config['model']['lesion_classifier']['epochs']):
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
                    predicted = (outputs > 0.5).float()
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.numpy())
            
            # Calculate metrics
            accuracy = accuracy_score(all_labels, all_preds)
            kappa = cohen_kappa_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average='weighted')
            
            # Save confusion matrix if best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                model_save_name = f"{model_name}_acc{accuracy:.3f}_epoch{epoch+1}.pth"
                best_model_path = os.path.join(model_dir, model_save_name)
                
                # Save confusion matrix
                cm_save_path = os.path.join(log_dir, f'confusion_matrix_epoch{epoch+1}.png')
                plot_confusion_matrix(all_labels, all_preds, 
                                    classes=train_dataset.labels,
                                    save_path=cm_save_path)
                
                # Save model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': best_accuracy,
                    'f1_score': f1,
                }, best_model_path)
                
                print(f'\nNew best model saved: {model_save_name}')
            
            print(f'Epoch [{epoch+1}/{config["model"]["lesion_classifier"]["epochs"]}]')
            print(f'Accuracy: {accuracy:.4f}, Kappa Score: {kappa:.4f}')
            print(f'F1-Score: {f1:.4f}')
            print('\nClassification Report:')
            print(classification_report(all_labels, all_preds))

if __name__ == '__main__':
    train_lesion()