import torch
import torch.nn as nn
import torchvision.models as models


class resnet101_lay4(nn.Module):
    """
    仅输出 ResNet101 layer4 特征图：[B, 2048, H/32, W/32]
    """

    def __init__(self, pretrained=True):
        super().__init__()
        resnet_backbone = models.resnet101(weights="IMAGENET1K_V1" if pretrained else None)
        self.feature_extractor = nn.Sequential(
            resnet_backbone.conv1,
            resnet_backbone.bn1,
            resnet_backbone.relu,
            resnet_backbone.maxpool,
            resnet_backbone.layer1,
            resnet_backbone.layer2,
            resnet_backbone.layer3,
            resnet_backbone.layer4,
        )

    def forward(self, x, return_features=False):
        features = self.feature_extractor(x)
        return features


if __name__ == "__main__":
    model = resnet101_lay4(pretrained=True).eval()
    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy_input)
    print(out.shape)

