
import torch.nn as nn
import torch.nn.functional as F
import math
class AdaptiveConvFeatureProjection_HP(nn.Module):
    """
    多尺度自适应卷积特征投影模块 (Hyper-Parameterized)

    输入: [B, C, H, W] (C, H, W可以是任意值)
    输出: [B, L, 512] 其中 L 由配置决定

    支持两种模式:
    1. 自适应模式 (adaptive): 调整到固定空间尺寸
    2. 保持模式 (keep): 保持输入的空间尺寸
    """

    def __init__(self, config):
        super().__init__()

        # ========== 从config读取参数 ==========
        self.in_channels = config.acfp_in_channels
        self.out_dim = 512  # 固定输出特征维度
        self.mode = config.acfp_mode  # 'adaptive' 或 'keep'

        # 设置目标序列长度
        if self.mode == 'adaptive':
            self.target_seq_len = config.acfp_target_seq_len
            # 计算最接近的目标空间尺寸
            self.target_spatial = self._calculate_spatial_from_seq_len(self.target_seq_len)
            actual_seq_len = self.target_spatial ** 2

            if actual_seq_len != self.target_seq_len:
                print(
                    f"[ACFP注意] 目标序列长度{self.target_seq_len}调整为{actual_seq_len}({self.target_spatial}x{self.target_spatial})")
        else:
            self.target_seq_len = None
            self.target_spatial = None

        self.use_residual = config.acfp_use_residual

        # # ========== 打印配置信息 ==========
        # print(f"[MultiScale-ACFP-HP] 配置:")
        # print(f"  输入通道: {self.in_channels}")
        # print(f"  输出维度: {self.out_dim}")
        # print(f"  模式: {self.mode}")
        # if self.mode == 'adaptive':
        #     print(f"  目标序列长度: {self.target_seq_len}")
        #     print(f"  目标空间尺寸: {self.target_spatial}x{self.target_spatial}")
        # print(f"  使用残差: {self.use_residual}")

        # ========== 1. 通道调整层 ==========
        self.channel_adjust = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_dim, kernel_size=1),
            nn.BatchNorm2d(self.out_dim),
            nn.GELU()
        )

        # ========== 2. 空间调整层（仅自适应模式需要） ==========
        if self.mode == 'adaptive':
            self.spatial_adjust = nn.AdaptiveAvgPool2d((self.target_spatial, self.target_spatial))

        # ========== 3. 空间特征提取层 ==========
        self.spatial_extractor = nn.Sequential(
            nn.Conv2d(self.out_dim, self.out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.out_dim),
            nn.GELU(),
            nn.Conv2d(self.out_dim, self.out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.out_dim),
            nn.GELU()
        )

        # ========== 4. 残差连接 ==========
        if self.use_residual:
            self.residual_adapter = nn.Conv2d(self.in_channels, self.out_dim, kernel_size=1)
        else:
            self.residual_adapter = None

        self._initialize_weights()

    def _calculate_spatial_from_seq_len(self, target_seq_len):
        """
        从目标序列长度计算最接近的空间尺寸

        Args:
            target_seq_len: 目标序列长度

        Returns:
            spatial_size: 最接近的空间尺寸（整数）
        """
        # 计算平方根
        sqrt_val = math.sqrt(target_seq_len)

        # 如果是完全平方数，直接返回
        if sqrt_val.is_integer():
            return int(sqrt_val)

        # 否则寻找最接近的完全平方数
        lower = int(math.floor(sqrt_val))
        upper = int(math.ceil(sqrt_val))

        # 选择最接近的完全平方数
        lower_sq = lower ** 2
        upper_sq = upper ** 2

        if (target_seq_len - lower_sq) <= (upper_sq - target_seq_len):
            return lower
        else:
            return upper

    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入特征 [batch_size, in_channels, H, W]

        Returns:
            output: 投影后的特征 [batch_size, L, 512]
        """
        B, C, H, W = x.shape

        # ========== 1. 通道验证 ==========
        if C != self.in_channels:
            print(f"[ACFP警告] 输入通道{C}≠期望{self.in_channels}")
            # 动态调整通道
            if not hasattr(self, 'dynamic_channel_adjust'):
                self.dynamic_channel_adjust = nn.Conv2d(C, self.in_channels, kernel_size=1)
                self.dynamic_channel_adjust = self.dynamic_channel_adjust.to(x.device)
            x = self.dynamic_channel_adjust(x)

        # 保存原始输入用于残差
        x_original = x

        # ========== 2. 通道调整 ==========
        x = self.channel_adjust(x)  # [B, 512, H, W]

        # ========== 3. 残差连接 ==========
        if self.use_residual and self.residual_adapter is not None:
            residual = self.residual_adapter(x_original)
            x = x + residual

        # ========== 4. 空间调整（自适应模式） ==========
        if self.mode == 'adaptive':
            x = self.spatial_adjust(x)  # [B, 512, target_spatial, target_spatial]
            H, W = self.target_spatial, self.target_spatial

        # ========== 5. 空间特征提取 ==========
        x = self.spatial_extractor(x)  # [B, 512, H, W]

        # ========== 6. 转换为序列格式 ==========
        # [B, 512, H, W] -> [B, H*W, 512]
        x = x.view(B, self.out_dim, -1).transpose(1, 2)

        return x
