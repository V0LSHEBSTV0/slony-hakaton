import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as T

class CustomResNetClassifier(nn.Module):
    """Классификатор на основе ResNet для изображений 512x512"""
    
    def __init__(self,path, num_classes=2, pretrained=True):
        super(CustomResNetClassifier, self).__init__()
        
        # Загружаем предобученную ResNet18
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # Убираем последний слой
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        
        # Добавляем адаптивные слои для работы с 512x512
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        self.load_state_dict(torch.load(path))
    
    def forward(self, x):
        # x имеет размер (batch_size, 3, 512, 512)
        features = self.resnet(x)  # (batch_size, 512, 16, 16)
        pooled = self.adaptive_pool(features)  # (batch_size, 512, 1, 1)
        flattened = pooled.view(pooled.size(0), -1)  # (batch_size, 512)
        output = self.classifier(flattened)  # (batch_size, num_classes)
        return output