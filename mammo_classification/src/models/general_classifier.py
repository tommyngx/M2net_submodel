import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights, ConvNeXt_Tiny_Weights, ConvNeXt_Base_Weights
import timm

class GeneralClassifier(nn.Module):
    def __init__(self, model_name='resnet50', num_classes=4, pretrained=True):
        super(GeneralClassifier, self).__init__()
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