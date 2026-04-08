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
# Model 1: 朴素视觉序列 + 朴素交叉注意力
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


# ---------------------------------------------------------------------------
# Model 6: 朴素双分支 + logits 相加（无 α、无 M）
# ---------------------------------------------------------------------------


class PlainSummationHead(nn.Module):
    """
    与 HMLH 相同的双分支结构，但输出 logits = image_logits + fusion_logits。
    对应文中 naive summation of dual-stream logits。
    """

    def __init__(self, config):
        super().__init__()
        self.intermediate_dim = config.hybrid_intermediate_dim
        self.num_classes = config.num_classes
        self.dropout_rate = config.hybrid_dropout_rate
        if config.hybrid_activation == "relu":
            self.act_fn = nn.ReLU()
        elif config.hybrid_activation == "gelu":
            self.act_fn = nn.GELU()
        elif config.hybrid_activation == "leaky_relu":
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


# ---------------------------------------------------------------------------
# 共享：堆叠交互块（Model 5 及可与 Model 1–4 对齐的 backbone）
# ---------------------------------------------------------------------------


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
    """每步：可选 MHCSRA → 视觉序列化（ACFP/Naive）→ 跨模态（Naive/MMAEF）→ 残差 + Norm；重复 stack_depth 次。"""

    def __init__(
        self,
        config,
        *,
        use_mhcsra: bool,
        use_acfp: bool,
        use_scmm: bool,
        stack_depth: int,
        img_channels: int = 2048,
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
            img, sem = self.norm(img, sem)
        return img, sem


# ---------------------------------------------------------------------------
# 各 Model 封装（forward 均返回 logits）
# ---------------------------------------------------------------------------


class PaperModel_1_Baseline(nn.Module):
    """Model 1：Backbone + 朴素序列 + 朴素交叉注意力 + Plain head。"""

    def __init__(self, label, config):
        super().__init__()
        self.config = config
        depth = _stack_depth(config)
        self.cnnbackbone = resnet101_lay4(pretrained=True)
        self.label = labal_to_enconde(label, config.device).unsqueeze(0).repeat(config.batch_size, 1, 1)
        self.backbone = StackedInteractionBackbone(
            config,
            use_mhcsra=False,
            use_acfp=False,
            use_scmm=False,
            stack_depth=depth,
            img_channels=2048,
        )
        self.head = PlainSummationHead(config)

    def forward(self, imgtensor: torch.Tensor, model_numbers=None) -> torch.Tensor:
        lab = self.label
        if lab.shape[0] != imgtensor.shape[0]:
            lab = self.label[: imgtensor.shape[0]]
        feat = self.cnnbackbone(imgtensor)
        img, sem = self.backbone(feat, lab)
        return self.head(img, sem)


class PaperModel_2_PlusACFP(nn.Module):
    """Model 2：在 Model 1 上用 ACFP 替代朴素展平。"""

    def __init__(self, label, config):
        super().__init__()
        self.config = config
        depth = _stack_depth(config)
        self.cnnbackbone = resnet101_lay4(pretrained=True)
        self.label = labal_to_enconde(label, config.device).unsqueeze(0).repeat(config.batch_size, 1, 1)
        self.backbone = StackedInteractionBackbone(
            config,
            use_mhcsra=False,
            use_acfp=True,
            use_scmm=False,
            stack_depth=depth,
            img_channels=2048,
        )
        self.head = PlainSummationHead(config)

    def forward(self, imgtensor: torch.Tensor, model_numbers=None) -> torch.Tensor:
        lab = self.label
        if lab.shape[0] != imgtensor.shape[0]:
            lab = self.label[: imgtensor.shape[0]]
        feat = self.cnnbackbone(imgtensor)
        img, sem = self.backbone(feat, lab)
        return self.head(img, sem)


class PaperModel_3_PlusSCMM(PaperModel_2_PlusACFP):
    """Model 3：在 Model 2 上用 SCMM（MMAEF_HP）替代朴素交叉注意力。"""

    def __init__(self, label, config):
        nn.Module.__init__(self)
        self.config = config
        depth = _stack_depth(config)
        self.cnnbackbone = resnet101_lay4(pretrained=True)
        self.label = labal_to_enconde(label, config.device).unsqueeze(0).repeat(config.batch_size, 1, 1)
        self.backbone = StackedInteractionBackbone(
            config,
            use_mhcsra=False,
            use_acfp=True,
            use_scmm=True,
            stack_depth=depth,
            img_channels=2048,
        )
        self.head = PlainSummationHead(config)


class PaperModel_4_PlusMHCSRA(PaperModel_3_PlusSCMM):
    """Model 4：在 Model 3 上于骨干后插入 MHCSRA。"""

    def __init__(self, label, config):
        nn.Module.__init__(self)
        self.config = config
        depth = _stack_depth(config)
        self.cnnbackbone = resnet101_lay4(pretrained=True)
        self.label = labal_to_enconde(label, config.device).unsqueeze(0).repeat(config.batch_size, 1, 1)
        ch = int(getattr(config, "mhcsra_out_channel", 2048))
        self.backbone = StackedInteractionBackbone(
            config,
            use_mhcsra=True,
            use_acfp=True,
            use_scmm=True,
            stack_depth=depth,
            img_channels=ch,
        )
        self.head = PlainSummationHead(config)


class PaperModel_5_Stacked(PaperModel_4_PlusMHCSRA):
    """Model 5：完整块 + 堆叠（堆叠次数 = max_iterations）。"""

    def __init__(self, label, config):
        PaperModel_4_PlusMHCSRA.__init__(self, label, config)


class PaperModel_6_PlainHeadBCE(nn.Module):
    """
    Model 6：以 Model 5 为特征骨干 + Plain head。
    BCE 在训练时选择；此处仅返回 logits。
    """

    def __init__(self, label, config):
        super().__init__()
        self._core = PaperModel_5_Stacked(label, config)

    def forward(self, imgtensor: torch.Tensor, model_numbers=None) -> torch.Tensor:
        return self._core(imgtensor, model_numbers)


class PaperModel_7_PlusASL(PaperModel_6_PlainHeadBCE):
    """Model 7：结构与 Model 6 相同；训练时用 ASL 代替 BCE。"""


class PaperModel_8_PlusAlphaGating(nn.Module):
    """Model 8：Model 5 特征 + HMLH（α 门控，默认不在语义支路用 M）。"""

    def __init__(self, label, config):
        super().__init__()
        self.config = config
        self.cnnbackbone = resnet101_lay4(pretrained=True)
        self.label = labal_to_enconde(label, config.device).unsqueeze(0).repeat(config.batch_size, 1, 1)
        ch = int(getattr(config, "mhcsra_out_channel", 2048))
        self.backbone = StackedInteractionBackbone(
            config,
            use_mhcsra=True,
            use_acfp=True,
            use_scmm=True,
            stack_depth=_stack_depth(config),
            img_channels=ch,
        )
        self.head = HybridMultiLabelHead_HP(config)

    def forward(self, imgtensor: torch.Tensor, model_numbers=None) -> torch.Tensor:
        lab = self.label
        if lab.shape[0] != imgtensor.shape[0]:
            lab = self.label[: imgtensor.shape[0]]
        feat = self.cnnbackbone(imgtensor)
        img, sem = self.backbone(feat, lab)
        return self.head(img, sem, use_matrix=False)


class PaperModel_9_PlusMMatrix(nn.Module):
    """Model 9：Model 5 特征 + HMLH 语义支路使用矩阵 M（无 α 门控时可用 image 权重趋近 0 近似；此处采用 use_matrix=True 且仍走完整 HMLH）。"""

    def __init__(self, label, config):
        super().__init__()
        self.config = config
        self.cnnbackbone = resnet101_lay4(pretrained=True)
        self.label = labal_to_enconde(label, config.device).unsqueeze(0).repeat(config.batch_size, 1, 1)
        ch = int(getattr(config, "mhcsra_out_channel", 2048))
        self.backbone = StackedInteractionBackbone(
            config,
            use_mhcsra=True,
            use_acfp=True,
            use_scmm=True,
            stack_depth=_stack_depth(config),
            img_channels=ch,
        )
        self.head = HybridMultiLabelHead_HP(config)
        with torch.no_grad():
            self.head.alpha.fill_(-20.0)

    def forward(self, imgtensor: torch.Tensor, model_numbers=None) -> torch.Tensor:
        lab = self.label
        if lab.shape[0] != imgtensor.shape[0]:
            lab = self.label[: imgtensor.shape[0]]
        feat = self.cnnbackbone(imgtensor)
        img, sem = self.backbone(feat, lab)
        return self.head(img, sem, use_matrix=True)


class MTI_HANet_Ours(nn.Module):
    """MTI-HANet：完整 HMLH（α + M）+ Model 5 级堆叠骨干。训练建议 ASL。"""

    def __init__(self, label, config):
        super().__init__()
        self.config = config
        self.cnnbackbone = resnet101_lay4(pretrained=True)
        self.label = labal_to_enconde(label, config.device).unsqueeze(0).repeat(config.batch_size, 1, 1)
        ch = int(getattr(config, "mhcsra_out_channel", 2048))
        self.backbone = StackedInteractionBackbone(
            config,
            use_mhcsra=True,
            use_acfp=True,
            use_scmm=True,
            stack_depth=_stack_depth(config),
            img_channels=ch,
        )
        self.head = HybridMultiLabelHead_HP(config)

    def forward(self, imgtensor: torch.Tensor, model_numbers=None) -> torch.Tensor:
        lab = self.label
        if lab.shape[0] != imgtensor.shape[0]:
            lab = self.label[: imgtensor.shape[0]]
        feat = self.cnnbackbone(imgtensor)
        img, sem = self.backbone(feat, lab)
        return self.head(img, sem, use_matrix=True)


class PaperModel_10_MMatrixAndAlphaGating(MTI_HANet_Ours):
    """
    Model 10 (+ M Matrix & α Gating):
    在 Model 6 同级的堆叠骨干（与 Model 5/6 一致：MHCSRA+ACFP+SCMM 堆叠）上，
    将分类头换为完整 HMLH：同时启用可学习标签相关矩阵 M 与自适应 logit 融合权重 α。
    论文设定下用标准 BCE 评估二者协同；训练脚本中通过 loss_kind='bce' 选用 BCEWithLogitsLoss。
    """

    pass


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------

_PAPER_MODEL_REGISTRY = {
    "model_1": PaperModel_1_Baseline,
    "model_2": PaperModel_2_PlusACFP,
    "model_3": PaperModel_3_PlusSCMM,
    "model_4": PaperModel_4_PlusMHCSRA,
    "model_5": PaperModel_5_Stacked,
    "model_6": PaperModel_6_PlainHeadBCE,
    "model_7": PaperModel_7_PlusASL,
    "model_8": PaperModel_8_PlusAlphaGating,
    "model_9": PaperModel_9_PlusMMatrix,
    "model_10": PaperModel_10_MMatrixAndAlphaGating,
    "mti_hanet": MTI_HANet_Ours,
    "ours": MTI_HANet_Ours,
}


def get_paper_ablation_model(variant: str, label, config):
    """variant: model_1 … model_10, mti_hanet / ours。堆叠 = config.max_iterations（默认 2）。"""
    key = str(variant).lower().strip()
    if key not in _PAPER_MODEL_REGISTRY:
        raise ValueError(f"Unknown paper variant: {variant}. Choose from {list(_PAPER_MODEL_REGISTRY.keys())}")
    return _PAPER_MODEL_REGISTRY[key](label, config)


__all__ = [
    "get_paper_ablation_model",
    "_stack_depth",
    "NaiveVisualSequence",
    "NaiveCrossAttentionFusion",
    "PlainSummationHead",
    "StackedInteractionBackbone",
    "PaperModel_1_Baseline",
    "PaperModel_5_Stacked",
    "PaperModel_10_MMatrixAndAlphaGating",
    "MTI_HANet_Ours",
]
