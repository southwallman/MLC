from dataclasses import dataclass
import math


@dataclass
class ModelHyperParams:
    """模型超参数配置"""
    # 基础参数
    num_classes: int = 8
    batch_size: int = 32
    device: str = "cpu"

    # ===== MHCSRA模块参数 =====
    mhcsra_num_heads: int = 8
    mhcsra_feature_dim: int = 2048
    mhcsra_out_channel: int = 2048
    mhcsra_use_shallow: bool = True
    mhcsra_shallow_layers: int = 2
    mhcsra_dropout: float = 0.1
    # 新增：CSRA 专用参数 (为了兼容 Optuna csra 搜索空间)
    mhcsra_input_dim: int = 2048
    mhcsra_lam: float = 0.1
    mhcsra_fusion_method: str = "concat"
    mhcsra_use_residual: bool = False

    # ===== ACFP模块参数 =====
    acfp_in_channels: int = 2048  # 添加这个字段
    acfp_mode: str = 'adaptive'  # 添加这个字段
    acfp_target_seq_len: int = 169  # 改为169以匹配候选值
    acfp_use_residual: bool = True

    # ===== MMAEF模块参数 =====
    mmaef_num_heads: int = 8  # 注意力头数，必须能被512整除
    mmaef_ffn_hidden_ratio: float = 4.0  # FFN隐藏层扩展比例
    mmaef_use_dropout: bool = True  # 是否使用Dropout
    mmaef_dropout_rate: float = 0.1  # Dropout率

    # ===== HybridMultiLabelHead模块参数 =====
    hybrid_intermediate_dim: int = 256  # 中间层维度
    hybrid_dropout_rate: float = 0.1  # Dropout率
    hybrid_activation: str = 'relu'  # 激活函数：relu, gelu, leaky_relu, tanh
    hybrid_alpha_init: float = 0.5  # alpha参数初始值

    # ===== 迭代精炼模块参数 =====
    max_iterations: int = 3  # 最大迭代次数
    use_iteration_residual: bool = True  # 是否使用迭代残差
    residual_frequency: int = 2  # 残差连接频率（每几次迭代添加残差）
    residual_factor: float = 0.3  # 残差权重
    use_early_exit: bool = True  # 是否使用早退机制
    early_exit_threshold: float = 0.001  # 早退阈值
    early_exit_patience: int = 2  # 早退耐心值（连续几次小于阈值才退出）
    use_weighted_fusion: bool = True  # 是否使用加权融合
    fusion_mode: str = 'weighted'  # 融合模式：'last', 'weighted', 'learnable'
    learnable_residual: bool = False  # 是否使用可学习的残差权重

    def __post_init__(self):
        """参数验证"""
        # 基础参数验证
        assert self.num_classes > 0, "类别数必须大于0"
        assert self.batch_size > 0, "批次大小必须大于0"

        # MHCSRA参数验证
        assert self.mhcsra_num_heads > 0, "注意力头数必须大于0"
        assert self.mhcsra_feature_dim >= 512, "特征维度不能小于512"
        assert self.mhcsra_out_channel > 0, "输出通道必须大于0"
        assert 0 <= self.mhcsra_dropout <= 0.5, "Dropout率必须在0-0.5之间"
        assert self.mhcsra_shallow_layers > 0, "浅层层数必须大于0"
        # CSRA 专用验证
        assert self.mhcsra_input_dim > 0, "MHCSRA输入维度必须大于0"
        assert 0.0 <= self.mhcsra_lam <= 1.0, "MHCSRA lambda 参数必须在 0.0-1.0 之间"
        assert self.mhcsra_fusion_method in ["concat", "sum",
                                             "attention"], f"不支持的融合方式: {self.mhcsra_fusion_method}"

        # ACFP参数验证
        assert self.acfp_in_channels > 0, "输入通道必须大于0"
        assert self.acfp_mode in ['adaptive', 'keep'], "acfp_mode必须是'adaptive'或'keep'"
        assert self.acfp_target_seq_len > 0, "目标序列长度必须大于0"

        # MMAEF参数验证
        assert self.mmaef_num_heads > 0, "MMAEF注意力头数必须大于0"
        assert 512 % self.mmaef_num_heads == 0, f"mmaef_num_heads({self.mmaef_num_heads}) 必须能整除512"
        assert self.mmaef_ffn_hidden_ratio > 0, "MMAEF FFN隐藏层扩展比例必须大于0"
        assert 0 <= self.mmaef_dropout_rate <= 1, "MMAEF Dropout率必须在0-1之间"

        # HybridHead参数验证
        assert self.hybrid_intermediate_dim > 0, "混合头中间维度必须大于0"
        assert 0 <= self.hybrid_dropout_rate <= 1, "混合头Dropout率必须在0-1之间"
        assert self.hybrid_activation in ['relu', 'gelu', 'leaky_relu', 'tanh'], \
            f"不支持的激活函数: {self.hybrid_activation}"
        assert 0 <= self.hybrid_alpha_init <= 1, "alpha初始值必须在0-1之间"

        # 迭代参数验证
        assert self.max_iterations > 0, "最大迭代次数必须大于0"
        assert 0 <= self.residual_factor <= 1, "残差权重必须在0-1之间"
        assert self.residual_frequency > 0, "残差连接频率必须大于0"
        assert self.early_exit_threshold > 0, "早退阈值必须大于0"
        assert self.early_exit_patience > 0, "早退耐心值必须大于0"
        assert self.fusion_mode in ['last', 'weighted', 'learnable'], "融合模式必须是'last'、'weighted'或'learnable'"

        # 验证是否为完全平方数
        sqrt_val = math.sqrt(self.acfp_target_seq_len)
        if not sqrt_val.is_integer():
            raise ValueError(f"acfp_target_seq_len必须是完全平方数，当前值: {self.acfp_target_seq_len}")

        # 设备检查
        if self.device == "cuda":
            import torch
            if not torch.cuda.is_available():
                self.device = "cpu"