import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix2(cm, classes, title='Confusion Matrix', cmap=None):
    import matplotlib.pyplot as plt
    import numpy as np

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    threshold = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > threshold else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def plot_roc_curve(fpr, tpr, title='Receiver Operating Characteristic'):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.show()



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


def plot_predictions(images, labels, predictions, class_names, save_path, num_samples=12):
    """Plot grid of images with ground truth and predicted labels"""
    import matplotlib.pyplot as plt
    import torch
    import numpy as np
    
    # Convert tensors to numpy arrays
    images = images.cpu().numpy()
    
    # Create grid plot
    fig, axes = plt.subplots(3, 4, figsize=(15, 12))
    axes = axes.ravel()
    
    for idx in range(num_samples):
        # Get single image and convert from CHW to HWC format
        img = np.transpose(images[idx], (1, 2, 0))
        
        # Denormalize image - for A.Normalize(mean=(0,0,0), std=(1,1,1))
        img = img * 255.0  # Scale back to [0,255] range
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        # Plot image
        axes[idx].imshow(img)
        axes[idx].axis('off')
        
        # Add ground truth and prediction labels
        true_label = class_names[labels[idx]]
        pred_label = class_names[predictions[idx]]
        color = 'green' if true_label == pred_label else 'red'
        
        axes[idx].set_title(f'True: {true_label}\nPred: {pred_label}', 
                           color=color, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()