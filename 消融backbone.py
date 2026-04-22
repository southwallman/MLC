"""
论文式消融模型：骨干网络 (Backbone) 替换实验。
对应论文中替换不同 CNN 提取器的消融实验表格。

- 保持 MTI_HANet 的完全体配置 (MHCSRA + ACFP + SCMM + HMLH)。
- 仅替换特征提取网络 (ResNet101 -> ResNet50, DenseNet121, GoogLeNet, VGG16_bn 等)。
- 关键：为了不破坏现有的交互模块 (2048通道) 维度设计，使用 1x1 卷积对齐通道数。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models

from HMLH_HP import HybridMultiLabelHead_HP
from tonguedx_MLC.testencode import labal_to_enconde

# 复用原消融文件中的堆叠骨干和工具函数
from paper_ablation_MTI_HANet import StackedInteractionBackbone, _stack_depth


# ---------------------------------------------------------------------------
# 通用骨干网络提取器 (带通道对齐功能)
# ---------------------------------------------------------------------------

class CNNBackboneExtractor(nn.Module):
    """
    统一的骨干网络特征提取器。
    自动剥离分类头与池化层，返回 [B, 2048, H, W] 特征图以无缝对接后续模块。
    """

    def __init__(self, backbone_type: str, pretrained: bool = True):
        super().__init__()
        self.backbone_type = backbone_type.lower()

        # 🚨 核心修复：将过时的 pretrained 转换为新的 weights 参数
        selected_weights = 'DEFAULT' if pretrained else None

        if self.backbone_type == 'resnet101':
            net = models.resnet101(weights=selected_weights)
            self.features = nn.Sequential(*list(net.children())[:-2])
            self.out_channels = 2048

        elif self.backbone_type == 'resnet50':
            net = models.resnet50(weights=selected_weights)
            self.features = nn.Sequential(*list(net.children())[:-2])
            self.out_channels = 2048

        elif self.backbone_type == 'resnet34':
            net = models.resnet34(weights=selected_weights)
            self.features = nn.Sequential(*list(net.children())[:-2])
            self.out_channels = 512

        elif self.backbone_type == 'densenet121':
            net = models.densenet121(weights=selected_weights)
            # DenseNet 的 features 出来后需要过一个 ReLU
            self.features = nn.Sequential(net.features, nn.ReLU(inplace=True))
            self.out_channels = 1024

        elif self.backbone_type == 'vgg16_bn':
            net = models.vgg16_bn(weights=selected_weights)
            self.features = net.features
            self.out_channels = 512

        elif self.backbone_type == 'googlenet':
            # GoogLeNet 需要关闭 transform_input 避免冲突
            net = models.googlenet(weights=selected_weights, transform_input=False)
            # 剥离最后的 avgpool, dropout, fc
            self.features = nn.Sequential(*list(net.children())[:-3])
            self.out_channels = 1024

        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_type}")

        # 通道对齐层：如果输出通道不是 2048，就用 1x1 卷积升维/降维，
        # 这样就不需要修改后面 MHCSRA 和 HMLH 里的 hardcode 维度了。
        if self.out_channels == 2048:
            self.align_conv = nn.Identity()
        else:
            self.align_conv = nn.Sequential(
                nn.Conv2d(self.out_channels, 2048, kernel_size=1, bias=False),
                nn.BatchNorm2d(2048),
                nn.ReLU(inplace=True)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.align_conv(x)  # 统一对齐到 [B, 2048, H, W]
        return x


# ---------------------------------------------------------------------------
# MTI-HANet 核心类 (可变 Backbone)
# ---------------------------------------------------------------------------

class MTI_HANet_AnyBackbone(nn.Module):
    """MTI-HANet 完全体，支持动态传入骨干网络类型。"""

    def __init__(self, label, config, backbone_type: str):
        super().__init__()
        self.config = config

        # 1. 替换为通用提取器
        self.cnnbackbone = CNNBackboneExtractor(backbone_type, pretrained=True)

        # 2. 标签编码
        self.label = labal_to_enconde(label, config.device).unsqueeze(0).repeat(config.batch_size, 1, 1)

        # 3. 核心交互网络 (参数与 MTI_HANet_Ours 完全一致)
        ch = int(getattr(config, "mhcsra_out_channel", 2048))
        self.backbone = StackedInteractionBackbone(
            config,
            use_mhcsra=True,
            use_acfp=True,
            use_scmm=True,
            stack_depth=_stack_depth(config),
            img_channels=ch,  # 由于有 align_conv，这里始终是 2048
        )

        # 4. 混合多标签头
        self.head = HybridMultiLabelHead_HP(config)

    def forward(self, imgtensor: torch.Tensor, model_numbers=None) -> torch.Tensor:
        lab = self.label
        if lab.shape[0] != imgtensor.shape[0]:
            lab = self.label[: imgtensor.shape[0]]

        feat = self.cnnbackbone(imgtensor)
        img, sem = self.backbone(feat, lab)
        return self.head(img, sem, use_matrix=True)


# ---------------------------------------------------------------------------
# 具体的各个消融网络实例
# ---------------------------------------------------------------------------

class MTI_HANet_ResNet101(MTI_HANet_AnyBackbone):
    """基准：完全体 (与原 mti_hanet_ours 等价)"""

    def __init__(self, label, config):
        super().__init__(label, config, backbone_type='resnet101')


class MTI_HANet_ResNet50(MTI_HANet_AnyBackbone):
    """消融 1：换用更轻量的 ResNet50"""

    def __init__(self, label, config):
        super().__init__(label, config, backbone_type='resnet50')


class MTI_HANet_ResNet34(MTI_HANet_AnyBackbone):
    """消融 2：换用 ResNet34"""

    def __init__(self, label, config):
        super().__init__(label, config, backbone_type='resnet34')


class MTI_HANet_DenseNet121(MTI_HANet_AnyBackbone):
    """消融 3：换用密集连接网络 DenseNet121"""

    def __init__(self, label, config):
        super().__init__(label, config, backbone_type='densenet121')


class MTI_HANet_VGG16(MTI_HANet_AnyBackbone):
    """消融 4：换用经典直筒网络 VGG16 (带 BN)"""

    def __init__(self, label, config):
        super().__init__(label, config, backbone_type='vgg16_bn')


class MTI_HANet_GoogLeNet(MTI_HANet_AnyBackbone):
    """消融 5：换用多尺度感受野网络 GoogLeNet (Inception v1)"""

    def __init__(self, label, config):
        super().__init__(label, config, backbone_type='googlenet')


# ---------------------------------------------------------------------------
# 工厂模式：根据字符串获取模型
# ---------------------------------------------------------------------------

_BACKBONE_ABLATION_REGISTRY = {
    "resnet101": MTI_HANet_ResNet101,
    "resnet50": MTI_HANet_ResNet50,
    "resnet34": MTI_HANet_ResNet34,
    "densenet121": MTI_HANet_DenseNet121,
    "vgg16": MTI_HANet_VGG16,
    "googlenet": MTI_HANet_GoogLeNet,
}


def get_backbone_ablation_model(variant: str, label, config):
    """
    variant: 'resnet101', 'resnet50', 'densenet121', 'vgg16', 'googlenet'
    """
    key = str(variant).lower().strip()
    if key not in _BACKBONE_ABLATION_REGISTRY:
        raise ValueError(f"Unknown backbone variant: {variant}. Choose from {list(_BACKBONE_ABLATION_REGISTRY.keys())}")
    return _BACKBONE_ABLATION_REGISTRY[key](label, config)


__all__ = [
    "get_backbone_ablation_model",
    "CNNBackboneExtractor",
    "MTI_HANet_AnyBackbone",
    "MTI_HANet_ResNet101",
    "MTI_HANet_ResNet50",
    "MTI_HANet_ResNet34",
    "MTI_HANet_DenseNet121",
    "MTI_HANet_VGG16",
    "MTI_HANet_GoogLeNet",
]