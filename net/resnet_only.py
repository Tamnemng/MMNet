import torch
import torch.nn as nn
import torchvision.models as models

class Model(nn.Module):
    def __init__(self, num_class=10, pretrained=True, **kwargs):
        super(Model, self).__init__()
        
        # Tải ResNet50 pretrained
        self.model = models.resnet50(pretrained=pretrained)
        
        # Thay đổi lớp fully connected cuối cùng
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_class)

    def forward(self, x):
        # x shape: [Batch, Channel, Height, Width]
        return self.model(x)