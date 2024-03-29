import torch
import torch.nn as nn
from dinov2.hub.backbones import dinov2_vits14_reg

class DinoV2(torch.nn.Module):
    
    def __init__(self, load_attention=False) -> None:
        super(DinoV2, self).__init__()
        if not load_attention:
            self.transformer = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
        else:
            # Follow this: https://gitlab.com/ziegleto-machine-learning/dino/-/tree/main/
            self.transformer = dinov2_vits14_reg()
        self.classifier = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Linear(256, 5) # 5 classes
        )

    def forward(self, x):
        x_dino = self.transformer(x)
        x_normalize = self.transformer.norm(x_dino)
        x_logit = self.classifier(x_normalize)
        return x_logit


