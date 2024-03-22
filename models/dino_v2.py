import torch
import torch.nn as nn


class DinoV2(torch.nn.Module):
    
    def __init__(self) -> None:
        super(DinoV2, self).__init__()
        self.transformer = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
        self.classifier = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Linear(256, 5) # 5 classes
        )

    def forward(self, x):
        x = self.transformer(x)
        x = self.transformer.norm(x)
        x = self.classifier(x)
        return x

