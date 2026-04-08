import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from MHCSRA_HP import MultiHeadClassSpecificFeatureEnhancement
from model.resnet101_lay4 import resnet101_lay4 as resnet101_lay4
from tonguedx_MLC.loss import UniversalHierarchicalLoss
from ACFP_HP import AdaptiveConvFeatureProjection_HP
from MMAEF_HP import MMAEF_HP
from HMLH_HP import HybridMultiLabelHead_HP
from tonguedx_MLC.testencode import labal_to_enconde
from tonguedx_MLC.ASL import AsymmetricLoss


class IterativeFeatureRefinement_HP(nn.Module):
    """迭代特征精炼模块 (超参化)"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 从配置读取迭代参数
        self.max_iterations = config.max_iterations
        self.use_residual = config.use_iteration_residual
        self.residual_factor = config.residual_factor
        self.residual_frequency = config.residual_frequency
        self.use_early_exit = config.use_early_exit
        self.early_exit_threshold = config.early_exit_threshold
        self.early_exit_patience = config.early_exit_patience
        self.use_weighted_fusion = config.use_weighted_fusion
        self.fusion_mode = config.fusion_mode
        # 旧版 MHCSRA_HP 输出通道
        self.img_bn = nn.BatchNorm2d(config.mhcsra_out_channel)
        self.fusion_norm = nn.LayerNorm(512)  # fusion_feature: [B, num_labels, 512]
        self.fusion_bn = nn.BatchNorm1d(512)  # 作用在 fusion_feature 的最后一维上

        # 为每轮精炼后的特征提供非线性
        self.img_activation = nn.ReLU(inplace=True)
        self.fusion_activation = nn.ReLU(inplace=True)

    def forward(self, mhcsra, acfp, mmaef, out_head, imgtensor, label_features):
        """
        迭代精炼前向传播

        Args:
            mhcsra: MHCSRA模块
            acfp: ACFP模块
            mmaef: MMAEF模块
            out_head: 输出分类头
            imgtensor: 输入图像特征 [B, C, H, W]
            label_features: 初始标签特征 [B, num_labels, 512]

        Returns:
            fusion_logits: 所有迭代的logits列表
            final_logits: 最终融合的logits
            actual_iterations: 实际执行的迭代次数
        """

        current_img = imgtensor
        current_fusion=label_features
        # 早退机制相关变量
        actual_iterations = 0

        for i in range(self.max_iterations):
            actual_iterations += 1
            # 1. 图像特征精炼
            temp_img=current_img
            temp_fusion=current_fusion
            current_img = mhcsra(current_img)
            # 2. 转换为序列特征
            img_features = acfp(current_img)
            current_img=temp_img+current_img
            current_fusion=mmaef(current_fusion, img_features)
            current_fusion=temp_fusion+current_fusion
            # 归一化 + 非线性
            current_fusion = self.fusion_norm(current_fusion)  # 输入形状 (B, num_labels, 512)
            B, L, D = current_fusion.shape
            current_fusion = self.fusion_bn(current_fusion.contiguous().view(B * L, D)).view(B, L, D)
            current_fusion = self.fusion_activation(current_fusion)

            current_img = self.img_bn(current_img)  # 输入 (B, out_channel, H, W)
            current_img = self.img_activation(current_img)

        return  current_img,current_fusion


class our_model(nn.Module):
    """完整模型 (超参化版本)"""

    def __init__(self, label, config):
        super().__init__()

        # 保存配置
        self.config = config

        # 基础模块
        self.cnnbackbone = resnet101_lay4(pretrained=True)
        self.mhcsra = MultiHeadClassSpecificFeatureEnhancement(config=config)
        self.acfp = AdaptiveConvFeatureProjection_HP(config)
        self.MMAEF = MMAEF_HP(config)
        self.HMLH = HybridMultiLabelHead_HP(config)

        # alpha 不冻结，保持可学习（由 hybrid_alpha_init 初始化）

        # 消融实验统一：Exp4 固定迭代次数=3
        config.max_iterations = 3

        # 迭代精炼模块
        self.iterative_refinement = IterativeFeatureRefinement_HP(config)

        # 标签编码
        self.label = labal_to_enconde(label, config.device)
        self.label = self.label.unsqueeze(0).repeat(config.batch_size, 1, 1)

        # print(f"[完整模型-HP] 初始化完成")
        # print(f"  类别数: {config.num_classes}")
        # print(f"  批次大小: {config.batch_size}")
        # print(f"  设备: {config.device}")

    def forward(self, imgtensor, model_numbers=None):
        """
        前向传播

        Args:
            imgtensor: 输入图像 [B, 3, 224, 224]
            model_numbers: 迭代次数（如果为None则使用config.max_iterations）

        Returns:
            fusion_logits: 所有迭代的logits列表
            final_logits: 最终融合的logits
        """
        # 获取标签特征
        label_features = self.label

        # 如果batch_size不匹配，调整label_features
        if label_features.shape[0] != imgtensor.shape[0]:
            label_features = self.label[:imgtensor.shape[0]]

        # CNN特征提取
        imgtensor_features = self.cnnbackbone(imgtensor)

        # 迭代精炼
        current_img,current_fusion = self.iterative_refinement(
            self.mhcsra, self.acfp, self.MMAEF, self.HMLH,
            imgtensor_features, label_features
        )
        out=self.HMLH(current_img, current_fusion)
        return out


def _fix_hmlh_alpha(hmlh: nn.Module, alpha_value: float = 0.6) -> None:
    """
    HMLH_HP 内部是：alpha = sigmoid(self.alpha)
    这里冻结 alpha 并将其设为给定的 alpha_value。
    """
    if not hasattr(hmlh, "alpha"):
        return
    alpha_value = float(alpha_value)
    # logit(p) = log(p/(1-p))
    logit = math.log(alpha_value / (1.0 - alpha_value))
    with torch.no_grad():
        hmlh.alpha.fill_(logit)
    hmlh.alpha.requires_grad_(False)


def _hmlh_image_logits(hmlh: HybridMultiLabelHead_HP, image_feature: torch.Tensor) -> torch.Tensor:
    """
    返回 image 分支 logits：[B, num_classes]
    """
    x = hmlh.image_pool(image_feature).flatten(1)  # [B, 2048]
    x = hmlh.image_bn(x)
    x = hmlh.image_fc1(x)
    x = hmlh.act_fn(x)
    x = hmlh.image_dropout(x)
    return hmlh.image_fc2(x)


def _hmlh_fusion_logits(
    hmlh: HybridMultiLabelHead_HP,
    fusion_feature: torch.Tensor,
    use_matrix: bool = False,
) -> torch.Tensor:
    """
    返回 fusion 分支 logits：[B, num_classes]
    fusion_feature: [B, num_labels, 512]
    """
    x = fusion_feature
    B, L, _ = x.shape
    for i, fc in enumerate(hmlh.fusion_fcs):
        x = fc(x)  # [B, L, next_dim]
        x = hmlh.fusion_bns[i](x.contiguous().view(B * L, -1)).view(B, L, -1)
        x = hmlh.act_fn(x)
        x = hmlh.fusion_dropout(x)
    x = hmlh.fusion_out(x).squeeze(-1)  # [B, L]
    if use_matrix:
        x = x @ hmlh.correlation_matrix
    return x


class _ResNet101ClipAblationBase(nn.Module):
    """
    消融实验基类（按模块开关重组）。
    - 输出 mode:
      - "image": 只输出 image 分支 logits
      - "fusion": 只输出 fusion 分支 logits
      - "both": 两分支融合（使用固定 alpha=0.6）
    """

    def __init__(
        self,
        label,
        config,
        *,
        use_backbone: bool,
        use_iterative: bool,
        use_mhcsra: bool,
        iter_steps: int,
        output_mode: str,
        use_correlation_matrix: bool = False,
        alpha_value: float = 0.6,
    ):
        super().__init__()

        assert output_mode in {"image", "fusion", "both"}
        self.use_backbone = use_backbone
        self.use_iterative = use_iterative
        self.use_mhcsra = use_mhcsra
        self.iter_steps = int(iter_steps)
        self.output_mode = output_mode
        self.use_correlation_matrix = use_correlation_matrix

        # backbone
        self.cnnbackbone = resnet101_lay4(pretrained=True) if use_backbone else None

        # label 编码
        self.label = labal_to_enconde(label, config.device)
        self.label = self.label.unsqueeze(0).repeat(config.batch_size, 1, 1)

        # head
        self.HMLH = HybridMultiLabelHead_HP(config)

        # iterative components（只在需要时创建）
        if use_iterative:
            self.mhcsra = MultiHeadClassSpecificFeatureEnhancement(config=config) if use_mhcsra else None
            self.acfp = AdaptiveConvFeatureProjection_HP(config)
            self.MMAEF = MMAEF_HP(config)

            # 为了复用你在 IterativeFeatureRefinement_HP 里新增的 BN/ReLU 层
            # 这里直接让 config.max_iterations 与当前消融设置一致
            config.max_iterations = self.iter_steps
            self.iterative_refinement = IterativeFeatureRefinement_HP(config)
        else:
            self.mhcsra = None
            self.acfp = None
            self.MMAEF = None
            self.iterative_refinement = None

    def forward(self, imgtensor, model_numbers=None):
        # label features: [B, num_labels, 512]
        label_features = self.label
        if label_features.shape[0] != imgtensor.shape[0]:
            label_features = self.label[: imgtensor.shape[0]]

        if self.use_backbone:
            imgtensor_features = self.cnnbackbone(imgtensor)
        else:
            imgtensor_features = None

        # 初始 current
        current_img = imgtensor_features
        current_fusion = label_features

        if self.use_iterative:
            assert self.acfp is not None and self.MMAEF is not None and self.iterative_refinement is not None
            for _ in range(self.iter_steps):
                temp_img = current_img
                temp_fusion = current_fusion

                if self.use_mhcsra:
                    current_img = self.mhcsra(current_img)
                    current_img = temp_img + current_img
                else:
                    # 不做 MHCSRA：不引入 residual 的加倍效应，直接保持原图像特征
                    current_img = temp_img

                # ACFP -> MMAEF
                img_features = self.acfp(current_img)
                current_fusion = self.MMAEF(current_fusion, img_features)
                current_fusion = temp_fusion + current_fusion

                # 复用你在 IterativeFeatureRefinement_HP 里新增的归一化/激活
                current_fusion = self.iterative_refinement.fusion_norm(current_fusion)
                B, L, D = current_fusion.shape
                current_fusion = (
                    self.iterative_refinement.fusion_bn(
                        current_fusion.contiguous().view(B * L, D)
                    ).view(B, L, D)
                )
                current_fusion = self.iterative_refinement.fusion_activation(current_fusion)

                current_img = self.iterative_refinement.img_bn(current_img)
                current_img = self.iterative_refinement.img_activation(current_img)

        # 输出 logits
        if self.output_mode == "image":
            assert current_img is not None, "image-only 模式需要 backbone 特征"
            return _hmlh_image_logits(self.HMLH, current_img)
        if self.output_mode == "fusion":
            return _hmlh_fusion_logits(
                self.HMLH,
                current_fusion,
                use_matrix=self.use_correlation_matrix,
            )

        # both: 走 HMLH 的融合（alpha 固定 0.6）
        return self.HMLH(current_img, current_fusion, use_matrix=self.use_correlation_matrix)


# ===================== 6 个消融模型（Exp0/1/2/3/5/6）=====================


class our_model_exp0_resnet101_imageonly(_ResNet101ClipAblationBase):
    """Exp0：只用 ResNet101 + HMLH image 分支（绕开迭代精炼）"""

    def __init__(self, label, config):
        super().__init__(
            label,
            config,
            use_backbone=True,
            use_iterative=False,
            use_mhcsra=False,
            iter_steps=0,
            output_mode="image",
            use_correlation_matrix=False,
        )


class our_model_exp1_resnet101_fusiononly(_ResNet101ClipAblationBase):
    """Exp1：只用 HMLH fusion 分支（不使用图像特征）"""

    def __init__(self, label, config):
        super().__init__(
            label,
            config,
            use_backbone=False,
            use_iterative=False,
            use_mhcsra=False,
            iter_steps=0,
            output_mode="fusion",
            use_correlation_matrix=False,
        )


class our_model_exp2_no_mhcsra_iter1(_ResNet101ClipAblationBase):
    """Exp2：迭代=1，跳过 MHCSRA，仅 ACFP + MMAEF"""

    def __init__(self, label, config):
        super().__init__(
            label,
            config,
            use_backbone=True,
            use_iterative=True,
            use_mhcsra=False,
            iter_steps=1,
            output_mode="both",
            use_correlation_matrix=False,
        )


class our_model_exp3_mhcsra_iter1(_ResNet101ClipAblationBase):
    """Exp3：迭代=1，启用 MHCSRA + ACFP + MMAEF"""

    def __init__(self, label, config):
        super().__init__(
            label,
            config,
            use_backbone=True,
            use_iterative=True,
            use_mhcsra=True,
            iter_steps=1,
            output_mode="both",
            use_correlation_matrix=False,
        )


class our_model_exp5_full_iter3_with_correlation(_ResNet101ClipAblationBase):
    """Exp5：完整链路迭代=3，启用 correlation_matrix"""

    def __init__(self, label, config):
        super().__init__(
            label,
            config,
            use_backbone=True,
            use_iterative=True,
            use_mhcsra=True,
            iter_steps=3,
            output_mode="both",
            use_correlation_matrix=True,
        )


class our_model_exp6_full_iter2(_ResNet101ClipAblationBase):
    """Exp6：完整链路（迭代步由 config.max_iterations 驱动），correlation_matrix 默认关闭"""

    def __init__(self, label, config):
        iter_steps = int(getattr(config, "max_iterations", 2))
        super().__init__(
            label,
            config,
            use_backbone=True,
            use_iterative=True,
            use_mhcsra=True,
            iter_steps=iter_steps,
            output_mode="both",
            use_correlation_matrix=False,
        )


def get_ablation_model(model_variant: str, label, config):
    """
    model_variant in {"exp0","exp1","exp2","exp3","exp4","exp5","exp6"}
    - exp4 对应当前文件的 our_model（完整链路默认）
    """
    model_variant = str(model_variant).lower()
    if model_variant == "exp0":
        return our_model_exp0_resnet101_imageonly(label, config)
    if model_variant == "exp1":
        return our_model_exp1_resnet101_fusiononly(label, config)
    if model_variant == "exp2":
        return our_model_exp2_no_mhcsra_iter1(label, config)
    if model_variant == "exp3":
        return our_model_exp3_mhcsra_iter1(label, config)
    if model_variant == "exp4":
        return our_model(label, config)
    if model_variant == "exp5":
        return our_model_exp5_full_iter3_with_correlation(label, config)
    if model_variant == "exp6":
        return our_model_exp6_full_iter2(label, config)
    raise ValueError(f"Unknown model_variant: {model_variant}")


def create_multi_label_one_hot(batch_size, num_classes):
    """创建多标签one-hot编码"""
    # 随机生成0-1矩阵，然后确保每行至少有一个1
    multi_hot = torch.randint(0, 2, (batch_size, num_classes)).float()
    # 确保每行至少有一个1
    for i in range(batch_size):
        if multi_hot[i].sum() == 0:
            multi_hot[i, torch.randint(0, num_classes, (1,))] = 1
    return multi_hot


def test_model_hp():
    """测试超参数化模型"""
    print("开始测试超参数化模型...")

    # 创建配置
    from model_config import ModelHyperParams

    label = ['TonguePale', 'TipSideRed', 'Spot', 'Ecchymosis', 'Crack', 'Toothmark', 'FurThick', 'FurYellow']
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = ModelHyperParams(
        # 基础参数
        num_classes=8,
        batch_size=4,
        device=device,

        # MHCSRA参数
        mhcsra_num_heads=8,
        mhcsra_feature_dim=2048,
        mhcsra_out_channel=2048,

        # ACFP参数
        acfp_mode='adaptive',
        acfp_target_seq_len=169,

        # MMAEF参数
        mmaef_num_heads=8,

        # HybridHead参数
        hybrid_intermediate_dim=256,

        # 迭代参数
        max_iterations=3,
        use_iteration_residual=True,
        residual_frequency=2,
        residual_factor=0.3,
        use_early_exit=True,
        early_exit_threshold=0.001,
        early_exit_patience=2,
        use_weighted_fusion=True,
        fusion_mode='weighted',
        learnable_residual=False
    )

    # 创建模型
    model = our_model(label, config)
    model = model.to(device)

    # 创建测试数据
    batch_size = 4
    imgtensor = torch.randn(batch_size, 3, 448, 448).to(device)

    # 前向传播
    logits = model(imgtensor)

    # 验证输出
    print(f"\n测试结果:")
    print(f"  迭代次数: 3")

    print(f"  最终融合logits形状: {logits.shape}")

    # 创建目标标签
    targets = create_multi_label_one_hot(batch_size, config.num_classes).to(device)

    # 计算损失
    asl_loss_fn = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
    loss_value = asl_loss_fn(logits, targets)

    print(f"\n损失计算:")
    print(f"  ASL损失值: {loss_value.item():.4f}")

    # 初始化通用层次损失
    loss_fn = UniversalHierarchicalLoss(total_epochs=100)

    # 构建损失字典
    losses_dict = {
        'iteration_refinement': loss_value
    }

    # 计算总损失
    train_loss = loss_fn.training_loss(losses_dict, epoch=10)
    print(f"  总训练损失: {train_loss.item():.4f}")

    print("\n✅ 测试通过!")


if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)

    # 运行测试
    test_model_hp()