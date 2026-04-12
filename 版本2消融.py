"""
论文式消融模型（模块组合与 4.5.1/4.5.2 对应；堆叠次数仅跟工程配置走）。

- 堆叠块数：一律用 config.max_iterations（与 best_params JSON 一致），默认 2。
- 标签语义：labal_to_enconde（512）；SCMM = MMAEF_HP；朴素交叉注意力 = 单层 Q=标签 / KV=视觉。
- BCE/ASL 在训练脚本里选；本文件只返回 logits。

ACFP 输入通道：无 MHCSRA 时 acfp_in_channels=2048；有 MHCSRA 时一般为 mhcsra_out_channel。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from MHCSRA_HP import MultiHeadClassSpecificFeatureEnhancement
from model.resnet101_lay4 import resnet101_lay4
from ACFP_HP import AdaptiveConvFeatureProjection_HP
from MMAEF_HP import MultiHeadCrossAttention_HP, MMAEF_HP
from HMLH_HP import HybridMultiLabelHead_HP
from tonguedx_MLC.testencode import labal_to_enconde


def _stack_depth(config) -> int:
    """交互块重复次数 = max_iterations（默认 2，对齐当前最佳参数 JSON）。"""
    return int(getattr(config, "max_iterations", 2))


# ---------------------------------------------------------------------------
# 基础组件模块
# ---------------------------------------------------------------------------

class NaiveVisualSequence(nn.Module):
    """
    将 [B, C, H, W] 展平为 [B, H*W, C]，再投影到 512。
    对应文中 spatial flatten / resize 为注意力输入序列。
    """

    def __init__(self, in_channels: int = 2048, out_dim: int = 512):
        super().__init__()
        self.proj = nn.Linear(in_channels, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()  # [B, H*W, C]
        return self.proj(x)  # [B, H*W, 512]


class NaiveCrossAttentionFusion(nn.Module):
    """
    单层交叉注意力：Q=标签语义，K/V=视觉 token；残差 + LayerNorm。
    对应文中 naive cross-attention。
    """

    def __init__(self, config):
        super().__init__()
        self.cross = MultiHeadCrossAttention_HP(config)
        self.norm = nn.LayerNorm(512)

    def forward(self, label_emb: torch.Tensor, visual_tokens: torch.Tensor) -> torch.Tensor:
        out = self.cross(label_emb, visual_tokens)
        return self.norm(label_emb + out)


class PlainSummationHead(nn.Module):
    """
    与 HMLH 相同的双分支结构，但输出 logits = image_logits + fusion_logits。
    对应文中 naive summation of dual-stream logits。
    """

    def __init__(self, config):
        super().__init__()
        self.intermediate_dim = getattr(config, "hybrid_intermediate_dim", 256)
        self.num_classes = config.num_classes
        self.dropout_rate = getattr(config, "hybrid_dropout_rate", 0.1)

        act = getattr(config, "hybrid_activation", "relu")
        if act == "relu":
            self.act_fn = nn.ReLU()
        elif act == "gelu":
            self.act_fn = nn.GELU()
        elif act == "leaky_relu":
            self.act_fn = nn.LeakyReLU(0.1)
        else:
            self.act_fn = nn.Tanh()

        self.image_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.image_bn = nn.BatchNorm1d(2048)
        self.image_fc1 = nn.Linear(2048, self.intermediate_dim)
        self.image_dropout = nn.Dropout(self.dropout_rate)
        self.image_fc2 = nn.Linear(self.intermediate_dim, self.num_classes)

        self.fusion_input_dim = self.num_classes * 512
        self.fusion_bn = nn.BatchNorm1d(self.fusion_input_dim)
        self.fusion_fc1 = nn.Linear(self.fusion_input_dim, self.intermediate_dim)
        self.fusion_dropout = nn.Dropout(self.dropout_rate)
        self.fusion_fc2 = nn.Linear(self.intermediate_dim, self.num_classes)

    def forward(self, image_feature: torch.Tensor, fusion_feature: torch.Tensor) -> torch.Tensor:
        x_img = self.image_pool(image_feature).flatten(1)
        x_img = self.image_bn(x_img)
        x_img = self.image_fc1(x_img)
        x_img = self.act_fn(x_img)
        x_img = self.image_dropout(x_img)
        image_logits = self.image_fc2(x_img)

        x_fus = fusion_feature.reshape(fusion_feature.shape[0], -1)
        x_fus = self.fusion_bn(x_fus)
        x_fus = self.fusion_fc1(x_fus)
        x_fus = self.act_fn(x_fus)
        x_fus = self.fusion_dropout(x_fus)
        fusion_logits = self.fusion_fc2(x_fus)

        return image_logits + fusion_logits


class _StackNorm(nn.Module):
    """迭代后对视觉 2D 与语义 1D 做 BN/LN + ReLU。"""

    def __init__(self, img_channels: int):
        super().__init__()
        self.fusion_norm = nn.LayerNorm(512)
        self.fusion_bn = nn.BatchNorm1d(512)
        self.img_bn = nn.BatchNorm2d(img_channels)
        self.img_act = nn.ReLU(inplace=True)
        self.fus_act = nn.ReLU(inplace=True)

    def forward(self, img_bchw: torch.Tensor, sem: torch.Tensor):
        sem = self.fusion_norm(sem)
        B, L, D = sem.shape
        sem = self.fusion_bn(sem.contiguous().view(B * L, D)).view(B, L, D)
        sem = self.fus_act(sem)
        img_bchw = self.img_bn(img_bchw)
        img_bchw = self.img_act(img_bchw)
        return img_bchw, sem


class StackedInteractionBackbone(nn.Module):
    """🚨 修复版骨干：不破坏未经 MHCSRA 处理的原始图像特征。"""

    def __init__(
            self, config, *, use_mhcsra: bool, use_acfp: bool, use_scmm: bool,
            stack_depth: int, img_channels: int = 2048,
    ):
        super().__init__()
        self.stack_depth = int(stack_depth)
        self.use_mhcsra = use_mhcsra
        self.use_acfp = use_acfp
        self.use_scmm = use_scmm

        self.mhcsra = MultiHeadClassSpecificFeatureEnhancement(config) if use_mhcsra else None
        self.acfp = AdaptiveConvFeatureProjection_HP(config) if use_acfp else None
        self.naive_vis = None if use_acfp else NaiveVisualSequence(in_channels=img_channels, out_dim=512)

        if use_scmm:
            self.interaction = MMAEF_HP(config)
        else:
            self.interaction = NaiveCrossAttentionFusion(config)

        self.norm = _StackNorm(img_channels=img_channels)

    def forward(self, img_bchw: torch.Tensor, label_emb: torch.Tensor):
        sem = label_emb
        img = img_bchw
        for _ in range(self.stack_depth):
            img_prev, sem_prev = img, sem

            if self.use_mhcsra:
                img = img_prev + self.mhcsra(img_prev)

            if self.use_acfp:
                tokens = self.acfp(img)
            else:
                tokens = self.naive_vis(img)

            sem = sem_prev + self.interaction(sem_prev, tokens)

            # 👇 核心修复：生成 norm 后的变量，但有条件地更新 img
            img_normed, sem = self.norm(img, sem)

            # 只有经过了 MHCSRA 修改的特征，才需要重新 BN，否则保留 ResNet 原厂特征！
            if self.use_mhcsra:
                img = img_normed

        return img, sem


# ---------------------------------------------------------------------------
# 🚨 严格对齐论文 Table X 与 Table 2 的消融模型 🚨
# ---------------------------------------------------------------------------

class PaperModel_1_Baseline(nn.Module):
    """【Table X】Model 1 (Base)：什么都不加 (False, False, False)"""

    def __init__(self, label, config):
        super().__init__()
        self.config = config
        self.cnnbackbone = resnet101_lay4(pretrained=True)
        self.label = labal_to_enconde(label, config.device).unsqueeze(0).repeat(config.batch_size, 1, 1)
        self.backbone = StackedInteractionBackbone(
            config, use_mhcsra=False, use_acfp=False, use_scmm=False,
            stack_depth=_stack_depth(config), img_channels=2048
        )
        self.head = PlainSummationHead(config)

    def forward(self, imgtensor: torch.Tensor, model_numbers=None) -> torch.Tensor:
        lab = self.label
        if lab.shape[0] != imgtensor.shape[0]:
            lab = self.label[: imgtensor.shape[0]]
        feat = self.cnnbackbone(imgtensor)
        img, sem = self.backbone(feat, lab)
        return self.head(img, sem)


class PaperModel_2_PlusMHCSRA(PaperModel_1_Baseline):
    """【Table X】Model 2：只加 MHCSRA (True, False, False)"""

    def __init__(self, label, config):
        super().__init__(label, config)
        ch = int(getattr(config, "mhcsra_out_channel", 2048))
        self.backbone = StackedInteractionBackbone(
            config, use_mhcsra=True, use_acfp=False, use_scmm=False,
            stack_depth=_stack_depth(config), img_channels=ch
        )


class PaperModel_3_MHCSRA_ACFP(PaperModel_1_Baseline):
    """【Table X】Model 3：加 MHCSRA + ACFP (True, True, False)"""

    def __init__(self, label, config):
        super().__init__(label, config)
        ch = int(getattr(config, "mhcsra_out_channel", 2048))
        self.backbone = StackedInteractionBackbone(
            config, use_mhcsra=True, use_acfp=True, use_scmm=False,
            stack_depth=_stack_depth(config), img_channels=ch
        )


class PaperModel_4_ACFP_SCMM(PaperModel_1_Baseline):
    """【Table X】Model 4：加 ACFP + SCMM (False, True, True)"""

    def __init__(self, label, config):
        super().__init__(label, config)
        self.backbone = StackedInteractionBackbone(
            config, use_mhcsra=False, use_acfp=True, use_scmm=True,
            stack_depth=_stack_depth(config), img_channels=2048
        )


class PaperModel_5_MHCSRA_SCMM(PaperModel_1_Baseline):
    """【Table X】Model 5：加 MHCSRA + SCMM (True, False, True)"""

    def __init__(self, label, config):
        super().__init__(label, config)
        ch = int(getattr(config, "mhcsra_out_channel", 2048))
        self.backbone = StackedInteractionBackbone(
            config, use_mhcsra=True, use_acfp=False, use_scmm=True,
            stack_depth=_stack_depth(config), img_channels=ch
        )


class PaperModel_6_PlainHeadBCE(PaperModel_1_Baseline):
    """【Table X】Model 6 (Ours Backbone)：三者全加 (True, True, True)。对应 Table 2 的 Base Head。"""

    def __init__(self, label, config):
        super().__init__(label, config)
        ch = int(getattr(config, "mhcsra_out_channel", 2048))
        self.backbone = StackedInteractionBackbone(
            config, use_mhcsra=True, use_acfp=True, use_scmm=True,
            stack_depth=_stack_depth(config), img_channels=ch
        )


class PaperModel_7_PlusASL(PaperModel_6_PlainHeadBCE):
    """【Table 2】Model 7：结构等同 Model 6，但在训练脚本里选用 ASL。"""
    pass


class PaperModel_8_PlusAlphaGating(PaperModel_1_Baseline):
    """【Table 2】Model 8：Model 6 特征 + 完整HMLH 但关闭 Matrix M (仅保留 α Gating)。"""

    def __init__(self, label, config):
        super().__init__(label, config)
        ch = int(getattr(config, "mhcsra_out_channel", 2048))
        self.backbone = StackedInteractionBackbone(
            config, use_mhcsra=True, use_acfp=True, use_scmm=True,
            stack_depth=_stack_depth(config), img_channels=ch
        )
        self.head = HybridMultiLabelHead_HP(config)

    def forward(self, imgtensor: torch.Tensor, model_numbers=None) -> torch.Tensor:
        lab = self.label[: imgtensor.shape[0]]
        feat = self.cnnbackbone(imgtensor)
        img, sem = self.backbone(feat, lab)
        return self.head(img, sem, use_matrix=False)  # 关闭 Matrix M


class PaperModel_9_PlusMMatrix(PaperModel_1_Baseline):
    """【Table 2】Model 9：Model 6 特征 + 完整HMLH 但强行关闭 α Gating (仅保留 Matrix M)。"""

    def __init__(self, label, config):
        super().__init__(label, config)
        ch = int(getattr(config, "mhcsra_out_channel", 2048))
        self.backbone = StackedInteractionBackbone(
            config, use_mhcsra=True, use_acfp=True, use_scmm=True,
            stack_depth=_stack_depth(config), img_channels=ch
        )
        self.head = HybridMultiLabelHead_HP(config)
        with torch.no_grad():
            self.head.alpha.fill_(-20.0)  # 强行填入极小值，关闭 α Gating

    def forward(self, imgtensor: torch.Tensor, model_numbers=None) -> torch.Tensor:
        lab = self.label[: imgtensor.shape[0]]
        feat = self.cnnbackbone(imgtensor)
        img, sem = self.backbone(feat, lab)
        return self.head(img, sem, use_matrix=True)


class PaperModel_10_MMatrixAndAlphaGating(PaperModel_1_Baseline):
    """【Table 2】Model 10：Model 6 特征 + 完整HMLH (Matrix M 和 α Gating 全开)。训练用 BCE。"""

    def __init__(self, label, config):
        super().__init__(label, config)
        ch = int(getattr(config, "mhcsra_out_channel", 2048))
        self.backbone = StackedInteractionBackbone(
            config, use_mhcsra=True, use_acfp=True, use_scmm=True,
            stack_depth=_stack_depth(config), img_channels=ch
        )
        self.head = HybridMultiLabelHead_HP(config)

    def forward(self, imgtensor: torch.Tensor, model_numbers=None) -> torch.Tensor:
        lab = self.label[: imgtensor.shape[0]]
        feat = self.cnnbackbone(imgtensor)
        img, sem = self.backbone(feat, lab)
        return self.head(img, sem, use_matrix=True)


class MTI_HANet_Ours(PaperModel_10_MMatrixAndAlphaGating):
    """【Table 2】MTI-HANet (Ours)：结构与 Model 10 完全一致，训练用 ASL。"""
    pass


# ---------------------------------------------------------------------------
# 工厂 (注意键名映射的更新)
# ---------------------------------------------------------------------------

_PAPER_MODEL_REGISTRY = {
    "model_1": PaperModel_1_Baseline,
    "model_2": PaperModel_2_PlusMHCSRA,
    "model_3": PaperModel_3_MHCSRA_ACFP,
    "model_4": PaperModel_4_ACFP_SCMM,
    "model_5": PaperModel_5_MHCSRA_SCMM,
    "model_6": PaperModel_6_PlainHeadBCE,
    "model_7": PaperModel_7_PlusASL,
    "model_8": PaperModel_8_PlusAlphaGating,
    "model_9": PaperModel_9_PlusMMatrix,
    "model_10": PaperModel_10_MMatrixAndAlphaGating,
    "mti_hanet": MTI_HANet_Ours,
    "ours": MTI_HANet_Ours,
}


def get_paper_ablation_model(variant: str, label, config):
    """variant: model_1 … model_10, mti_hanet / ours。"""
    key = str(variant).lower().strip()
    if key not in _PAPER_MODEL_REGISTRY:
        raise ValueError(f"Unknown paper variant: {variant}. Choose from {list(_PAPER_MODEL_REGISTRY.keys())}")
    return _PAPER_MODEL_REGISTRY[key](label, config)


__all__ = [
    "get_paper_ablation_model",
    "_stack_depth",
]