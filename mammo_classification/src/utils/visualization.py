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


def plot_predictions(images, labels, predictions, class_names, save_path, filenames=None, num_samples=12):
    """
    Plot grid of images with filename, ground truth and predicted labels
    Args:
        images: tensor of images (normalized)
        labels: ground truth labels
        predictions: predicted labels 
        class_names: list of class names
        save_path: path to save plot
        filenames: list of image filenames
        num_samples: number of images to plot
    """
    import matplotlib.pyplot as plt
    import torch
    import numpy as np
    
    # Convert to numpy and move to CPU if needed
    images = images.cpu().numpy()
    
    # Create grid plot
    fig, axes = plt.subplots(3, 4, figsize=(16, 14))
    axes = axes.ravel()
    
    for idx in range(num_samples):
        # Get single image and convert from CHW to HWC format
        img = np.transpose(images[idx], (1, 2, 0))
        
        # Denormalize image if it was normalized with mean=(0,0,0), std=(1,1,1)
        img = (img * 255).astype(np.uint8)  # Simple denormalization
        
        # Plot image
        axes[idx].imshow(img)
        axes[idx].axis('off')
        
        # Add labels
        true_label = class_names[labels[idx]]
        pred_label = class_names[predictions[idx]]
        color = 'green' if true_label == pred_label else 'red'
        
        # Create title with filename if provided
        title = f'True: {true_label}\nPred: {pred_label}'
        if filenames is not None:
            filename = os.path.basename(filenames[idx])
            title = f'{filename}\n{title}'
        
        axes[idx].set_title(title, color=color, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()