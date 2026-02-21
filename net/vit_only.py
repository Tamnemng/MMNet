import torch
import torch.nn as nn
import torchvision.models as models


class Model(nn.Module):
    def __init__(self, num_class=10, pretrained=True, **kwargs):
        super(Model, self).__init__()

        # Tải ViT-B/16 pretrained trên ImageNet
        if pretrained:
            self.model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            self.model = models.vit_b_16(weights=None)

        # Thay đổi lớp classification head cuối cùng
        # ViT-B/16 có hidden_dim = 768
        self.model.heads.head = nn.Linear(768, num_class)

    def forward(self, x):
        # x shape: [Batch, Channel, Height, Width] (224x224)
        return self.model(x)
