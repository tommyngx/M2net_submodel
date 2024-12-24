import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights, ConvNeXt_Tiny_Weights, ConvNeXt_Base_Weights
import timm

class BiradsClassifier(nn.Module):
    def __init__(self, model_name='resnet50', num_classes=4, pretrained=True):
        super(BiradsClassifier, self).__init__()
        self.model_name = model_name
        
        if 'resnet' in model_name:
            self.model = self._init_resnet(model_name, num_classes, pretrained)
        elif 'efficientnet' in model_name:
            self.model = self._init_efficientnet(model_name, num_classes, pretrained)
        elif 'vit' in model_name:
            self.model = self._init_vit(num_classes, pretrained)
        elif 'convnext' in model_name:
            self.model = self._init_convnext(model_name, num_classes, pretrained)
        else:
            raise ValueError(f"Model {model_name} not supported")

    def _init_resnet(self, model_name, num_classes, pretrained):
        if model_name == 'resnet50':
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.resnet50(weights=weights)
        elif model_name == 'resnet101':
            model = models.resnet101(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    def _init_efficientnet(self, model_name, num_classes, pretrained):
        if model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=pretrained)
        elif model_name == 'efficientnet_b4':
            model = models.efficientnet_b4(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    def _init_vit(self, num_classes, pretrained):
        model = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
        model.head = nn.Linear(model.head.in_features, num_classes)
        return model

    def _init_convnext(self, model_name, num_classes, pretrained):
        if model_name == 'convnext_tiny':
            weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.convnext_tiny(weights=weights)
        elif model_name == 'convnext_base':
            weights = ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.convnext_base(weights=weights)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        return model

    def forward(self, x):
        return self.model(x)

    def train(self, mode=True):
        """
        Override train method to match PyTorch's expected behavior
        """
        return super().train(mode)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            output = self(x)
            _, predicted = torch.max(output.data, 1)
        return predicted

    def evaluate(self, test_loader, criterion, device):
        self.eval()
        total_loss = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.cpu().numpy())
                targets.extend(labels.cpu().numpy())
                
        return total_loss / len(test_loader), predictions, targets