
import torch
import torch.nn as nn
import timm

class ConvNeXtClassifier(nn.Module):

    def __init__(self, num_classes=2):

        super().__init__()

        self.backbone = timm.create_model(
            "convnext_base",
            pretrained=True,
            num_classes=0
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512,num_classes)
        )

    def forward(self,x):

        features = self.backbone(x)
        output = self.classifier(features)

        return output
