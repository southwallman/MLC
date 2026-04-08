import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn


class HybridMultiLabelHead_HP(nn.Module):
    """
    论文最终版：混合多标签分类头 (Hybrid Multi-Label Head)
    包含：视觉双分支特征提取、标签相关性矩阵 M、自适应门控 Alpha
    """

    def __init__(self, config):
        super().__init__()
        self.num_classes = config.num_classes
        self.intermediate_dim = getattr(config, 'hybrid_intermediate_dim', 512)
        self.dropout_rate = getattr(config, 'hybrid_dropout_rate', 0.5)

        # 激活函数选择
        activation_type = getattr(config, 'hybrid_activation', 'relu')
        if activation_type == "relu":
            self.act_fn = nn.ReLU()
        elif activation_type == "gelu":
            self.act_fn = nn.GELU()
        elif activation_type == "leaky_relu":
            self.act_fn = nn.LeakyReLU(0.1)
        else:
            self.act_fn = nn.Tanh()

        # ---------------- 视觉特征支路 (Image Branch) ----------------
        self.image_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.image_bn = nn.BatchNorm1d(2048)
        self.image_fc1 = nn.Linear(2048, self.intermediate_dim)
        self.image_dropout = nn.Dropout(self.dropout_rate)
        self.image_fc2 = nn.Linear(self.intermediate_dim, self.num_classes)

        # ---------------- 语义融合特征支路 (Fusion Branch) ----------------
        self.fusion_input_dim = self.num_classes * 512
        self.fusion_bn = nn.BatchNorm1d(self.fusion_input_dim)
        self.fusion_fc1 = nn.Linear(self.fusion_input_dim, self.intermediate_dim)
        self.fusion_dropout = nn.Dropout(self.dropout_rate)
        self.fusion_fc2 = nn.Linear(self.intermediate_dim, self.num_classes)

        # ---------------- 核心创新组件 ----------------
        # 标签相关性矩阵 M (Correlation Matrix)
        self.correlation_matrix = nn.Parameter(torch.eye(self.num_classes))

        # 自适应门控权重 Alpha (Adaptive Gating)
        # self.alpha = nn.Parameter(torch.zeros(self.num_classes))
        # 修复后的版本（标量，完美匹配你 .pth 里的 torch.Size([])）：
        self.alpha = nn.Parameter(torch.tensor(0.0))
    def forward(self, image_feature: torch.Tensor, fusion_feature: torch.Tensor,
                use_matrix: bool = True) -> torch.Tensor:
        # 1. 视觉支路前向
        x_img = self.image_pool(image_feature).flatten(1)
        x_img = self.image_bn(x_img)
        x_img = self.image_fc1(x_img)
        x_img = self.act_fn(x_img)
        x_img = self.image_dropout(x_img)
        image_logits = self.image_fc2(x_img)

        # 2. 语义融合支路前向
        x_fus = fusion_feature.reshape(fusion_feature.shape[0], -1)
        x_fus = self.fusion_bn(x_fus)
        x_fus = self.fusion_fc1(x_fus)
        x_fus = self.act_fn(x_fus)
        x_fus = self.fusion_dropout(x_fus)

        # 3. 判断是否启用矩阵 M (消融实验控制)
        if use_matrix:
            fusion_base = self.fusion_fc2(x_fus)
            # 经过矩阵 M 增强标签间的关联性
            fusion_logits = torch.matmul(fusion_base, self.correlation_matrix)
        else:
            fusion_logits = self.fusion_fc2(x_fus)

        # 4. 自适应门控 Alpha 加权融合
        alpha_weight = torch.sigmoid(self.alpha)
        final_logits = alpha_weight * image_logits + (1 - alpha_weight) * fusion_logits

        return final_logits


def test_hybrid_head_hp():
    """测试HybridMultiLabelHead-HP模块"""
    print("开始测试HybridMultiLabelHead-HP模块...")

    # 创建测试配置
    from dataclasses import dataclass

    @dataclass
    class TestConfig:
        num_classes: int = 8
        hybrid_intermediate_dim: int = 256
        hybrid_dropout_rate: float = 0.1
        hybrid_activation: str = 'relu'
        # hybrid_alpha_init 这个参数没用到，删掉了也不影响

    config = TestConfig()

    # 参数设置
    batch_size = 4

    # 🚨 修正 1：正确创建两个输入分支的数据
    # 视觉特征：通常是骨干网络 (如ResNet101) 出来的特征图 [B, 2048, H, W]
    image_x = torch.randn(batch_size, 2048, 7, 7)

    # 语义特征：多头交叉注意力出来的特征 [B, num_classes, 512]
    fusion_x = torch.randn(batch_size, config.num_classes, 512)

    print(f"创建测试数据:")
    print(f"  视觉输入 image_x: {image_x.shape}")
    print(f"  语义输入 fusion_x: {fusion_x.shape}")

    # 创建模型
    model = HybridMultiLabelHead_HP(config)

    # 前向传播
    with torch.no_grad():
        # 🚨 修正 2：必须同时传入 image_x 和 fusion_x
        output = model(image_feature=image_x, fusion_feature=fusion_x)

    # 验证输出形状
    expected_shape = (batch_size, config.num_classes)
    assert output.shape == expected_shape, f"输出形状{output.shape}应与{expected_shape}一致"
    print(f"\n✅ 测试通过! 输出形状: {output.shape}")

    # 🚨 修正 3：直接读取模型的 alpha 参数，并经过 sigmoid 激活（因为前向传播里是这么做的）
    with torch.no_grad():
        actual_alpha_weights = torch.sigmoid(model.alpha)

    print(f"当前自适应门控 Alpha 权重 (各类别视觉特征的占比):")
    # 打印格式化，保留3位小数
    print([round(val.item(), 3) for val in actual_alpha_weights])


if __name__ == "__main__":
    # 设置详细输出
    torch.manual_seed(42)

    # 运行测试
    test_hybrid_head_hp()