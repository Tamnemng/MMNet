import torch
import torch.nn as nn
import torchvision.models as models


class Model(nn.Module):
    def __init__(self, num_class=10, pretrained=True, freeze_backbone=True, **kwargs):
        super(Model, self).__init__()

        # Tải ViT-B/16 pretrained trên ImageNet
        if pretrained:
            self.model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            self.model = models.vit_b_16(weights=None)

        # Thay đổi lớp classification head cuối cùng
        # ViT-B/16 có hidden_dim = 768
        self.model.heads.head = nn.Linear(768, num_class)

        # Freeze backbone: chỉ train classification head
        # Rất quan trọng với dataset nhỏ để tránh phá hỏng feature pretrained
        if freeze_backbone and pretrained:
            for name, param in self.model.named_parameters():
                if 'heads.head' not in name:
                    param.requires_grad = False
            print(f"[ViT] Đã freeze backbone, chỉ train classification head "
                  f"({sum(p.numel() for p in self.model.parameters() if p.requires_grad)} params trainable)")

    def forward(self, x):
        # x shape: [Batch, Channel, Height, Width] (224x224)
        return self.model(x)
