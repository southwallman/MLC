import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadClassSpecificFeatureEnhancement(nn.Module):
    """
    多头类特定特征增强模块
    输入: [batch_size, in_channels, H, W]
    输出: [batch_size, out_channel, H, W]
    """

    def __init__(self, config):
        super().__init__()

        # ========== 从config读取参数 ==========
        # 可调参数
        self.num_classes = config.num_classes
        self.num_heads = config.mhcsra_num_heads
        self.feature_dim = config.mhcsra_feature_dim
        self.out_channel = config.mhcsra_out_channel

        # 新增：从config读取输入通道
        self.in_channels = getattr(config, 'mhcsra_in_channels', 2048)

        # 可选参数
        self.use_shallow = getattr(config, 'mhcsra_use_shallow', True)
        self.shallow_layers = getattr(config, 'mhcsra_shallow_layers', 2)
        self.dropout_rate = getattr(config, 'mhcsra_dropout', 0.0)

        # ========== 1. 浅层特征提取 ==========
        if self.use_shallow:
            shallow_layers = []
            in_ch = self.in_channels

            for i in range(self.shallow_layers):
                out_ch = self.feature_dim
                shallow_layers.extend([
                    nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                ])

                if self.dropout_rate > 0 and i < self.shallow_layers - 1:
                    shallow_layers.append(nn.Dropout2d(self.dropout_rate))

                in_ch = out_ch

            self.shallow_features = nn.Sequential(*shallow_layers)
        else:
            # 不使用浅层特征提取，直接调整通道
            self.shallow_features = nn.Conv2d(self.in_channels, self.feature_dim, 1)

        # ========== 2. 特征投影层 ==========
        self.feature_projection = nn.Conv2d(self.feature_dim, self.feature_dim, 1)

        # ========== 3. 类特定权重矩阵 ==========
        # 改为使用更稳定的初始化
        self.class_specific_weights = nn.Parameter(
            torch.randn(self.num_classes, self.num_heads, self.feature_dim) * 0.01
        )

        # ========== 4. 注意力融合层 ==========
        self.attention_fusion = nn.Sequential(
            nn.Conv2d(
                self.num_heads * self.feature_dim,
                out_channels=self.out_channel,
                kernel_size=1
            ),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(inplace=True)
        )

        # ========== 5. 残差连接适配层 ==========
        if self.in_channels != self.out_channel:
            self.residual_adapter = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channel, 1),
                nn.BatchNorm2d(self.out_channel)
            )
        else:
            self.residual_adapter = nn.Identity()

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # 类特定权重的初始化
        nn.init.xavier_normal_(self.class_specific_weights)

    def forward(self, x):
        """
        前向传播
        输入: x [batch_size, in_channels, H, W]
        输出: enhanced_x [batch_size, out_channel, H, W]
        """
        batch_size, C, H, W = x.shape
        spatial_size = H * W

        # 保存原始输入用于残差连接
        x_original = x

        # ========== 步骤1: 提取特征 ==========
        features = self.shallow_features(x)  # [B, feature_dim, H, W]

        # ========== 步骤2: 特征投影 ==========
        U = self.feature_projection(features)  # [B, feature_dim, H, W]

        # ========== 步骤3: 生成多头类特定注意力图 ==========
        # 重塑特征用于矩阵乘法
        U_reshaped = U.view(batch_size, self.feature_dim, -1).transpose(1, 2)  # [B, H*W, feature_dim]

        # 重塑类特定权重
        W_cs_reshaped = self.class_specific_weights.view(self.num_classes * self.num_heads,
                                                         self.feature_dim)  # [num_classes*num_heads, feature_dim]
        W_cs_reshaped = W_cs_reshaped.transpose(0, 1)  # [feature_dim, num_classes*num_heads]

        # 计算注意力分数
        attention_scores = torch.matmul(U_reshaped, W_cs_reshaped)  # [B, H*W, num_classes*num_heads]
        attention_scores = attention_scores / (self.feature_dim ** 0.5)

        # 重塑注意力分数
        attention_scores = attention_scores.view(batch_size, spatial_size, self.num_classes, self.num_heads)
        attention_scores = attention_scores.permute(0, 2, 3, 1).contiguous()  # [B, num_classes, num_heads, H*W]

        # Softmax归一化
        attention_weights = F.softmax(attention_scores, dim=-1)  # [B, num_classes, num_heads, H*W]
        attention_weights = attention_weights.view(batch_size, self.num_classes, self.num_heads, H,
                                                   W)  # [B, num_classes, num_heads, H, W]

        # ========== 步骤4: 应用注意力并生成增强特征 ==========
        # 更高效的处理方式：使用广播和矩阵乘法
        # 扩展特征维度以便广播
        features_exp = features.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, feature_dim, H, W]

        # 扩展注意力权重维度
        attention_weights_exp = attention_weights.unsqueeze(3)  # [B, num_classes, num_heads, 1, H, W]

        # 应用注意力：广播相乘
        weighted_features = features_exp * attention_weights_exp  # [B, num_classes, num_heads, feature_dim, H, W]

        # 重塑用于融合
        weighted_features = weighted_features.view(
            batch_size * self.num_classes,
            self.num_heads * self.feature_dim,
            H, W
        )  # [B*num_classes, num_heads*feature_dim, H, W]

        # 融合特征
        class_outputs = self.attention_fusion(weighted_features)  # [B*num_classes, out_channel, H, W]

        # 重塑回原始维度
        class_outputs = class_outputs.view(
            batch_size, self.num_classes, self.out_channel, H, W
        )  # [B, num_classes, out_channel, H, W]

        # ========== 步骤5: 融合所有类别的增强特征 ==========
        # 对所有类别取平均
        final_output = class_outputs.mean(dim=1)  # [B, out_channel, H, W]

        # ========== 步骤6: 残差连接 ==========
        x_original = self.residual_adapter(x_original)
        residual_output = final_output + x_original  # [B, out_channel, H, W]

        return residual_output

    def get_attention_maps(self, x):
        """
        获取注意力图（用于可视化和分析）
        返回所有类别的注意力图，对所有注意力头取平均

        Args:
            x: 输入特征 [batch_size, in_channels, H, W]

        Returns:
            attention_maps: [batch_size, num_classes, H, W]
        """
        batch_size, C, H, W = x.shape
        spatial_size = H * W

        # 提取特征
        features = self.shallow_features(x)

        # 特征投影
        U = self.feature_projection(features)

        # 计算注意力
        U_reshaped = U.view(batch_size, self.feature_dim, -1).transpose(1, 2)
        W_cs_reshaped = self.class_specific_weights.view(self.num_classes * self.num_heads, self.feature_dim)
        W_cs_reshaped = W_cs_reshaped.transpose(0, 1)

        # 注意力分数
        attention_scores = torch.matmul(U_reshaped, W_cs_reshaped) / (self.feature_dim ** 0.5)

        # 重塑和softmax
        attention_scores = attention_scores.view(batch_size, spatial_size, self.num_classes, self.num_heads)
        attention_scores = attention_scores.permute(0, 2, 3, 1).contiguous()
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = attention_weights.view(batch_size, self.num_classes, self.num_heads, H, W)

        # 对所有注意力头取平均
        attention_maps = attention_weights.mean(dim=2)  # [batch_size, num_classes, H, W]

        return attention_maps


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 假设ModelHyperParams已定义
    try:
        from model_config import ModelHyperParams
    except ImportError:
        # 创建一个简单的配置类用于测试
        class ModelHyperParams:
            def __init__(self):
                self.num_classes = 8
                self.mhcsra_num_heads = 12
                self.mhcsra_feature_dim = 1024
                self.mhcsra_out_channel = 128
                self.mhcsra_dropout = 0.2
                self.mhcsra_use_shallow = True
                self.mhcsra_shallow_layers = 2
                self.mhcsra_in_channels = 2048

    # 创建配置
    config = ModelHyperParams()

    # 创建模块
    model = MultiHeadClassSpecificFeatureEnhancement(config)

    # 测试输入
    batch_size = 4
    dummy_input = torch.randn(batch_size, 2048, 7, 7)

    # 前向传播
    output = model(dummy_input)
    print(f"\n输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")

    # 获取注意力图
    attention_maps = model.get_attention_maps(dummy_input)
    print(f"注意力图形状: {attention_maps.shape}")  # [4, 8, 7, 7]