import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention_HP(nn.Module):
    """多头自注意力机制 (超参化) - 固定维度512"""

    def __init__(self, config):
        super().__init__()
        self.dim = 512
        self.num_heads = config.mmaef_num_heads
        self.head_dim = self.dim // self.num_heads
        assert self.dim % self.num_heads == 0, f"dim {self.dim} 必须能被 num_heads {self.num_heads} 整除"
        self.qkv = nn.Linear(self.dim, 3 * self.dim)
        self.proj = nn.Linear(self.dim, self.dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class MultiHeadCrossAttention_HP(nn.Module):
    """多头交叉注意力机制 (超参化) - 固定维度512"""

    def __init__(self, config):
        super().__init__()
        self.dim = 512
        self.num_heads = config.mmaef_num_heads
        self.head_dim = self.dim // self.num_heads
        assert self.dim % self.num_heads == 0, f"dim {self.dim} 必须能被 num_heads {self.num_heads} 整除"
        self.q_proj = nn.Linear(self.dim, self.dim)
        self.kv_proj = nn.Linear(self.dim, 2 * self.dim)
        self.proj = nn.Linear(self.dim, self.dim)

    def forward(self, x, context):
        B, N, C = x.shape
        _, N_c, _ = context.shape
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(context).reshape(B, N_c, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class MSSModule_HP(nn.Module):
    """MSS模块 (超参化) - 固定维度512"""

    def __init__(self, config):
        super().__init__()
        self.dim = 512
        self.num_heads = config.mmaef_num_heads
        self.head_dim = self.dim // self.num_heads
        assert self.dim % self.num_heads == 0, f"dim {self.dim} 必须能被 num_heads {self.num_heads} 整除"
        self.qk = nn.Linear(self.dim, 2 * self.dim)
        self.proj = nn.Linear(self.dim, self.dim)

    def forward(self, x):
        B, N, C = x.shape
        qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]
        attn_scores = q @ k.transpose(-2, -1)
        attn_scores = attn_scores * (self.head_dim ** -0.5)
        attn_weights = attn_scores.softmax(dim=-1)
        v = x.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        attn_output = attn_weights @ v
        attn_output = attn_output.transpose(1, 2).reshape(B, N, C)
        x_attn = torch.sigmoid(attn_output)
        return self.proj(x_attn)


class FeedForwardNetwork_HP(nn.Module):
    """前馈神经网络 (超参化) - 固定维度512"""

    def __init__(self, config):
        super().__init__()
        self.dim = 512
        hidden_dim = int(self.dim * config.mmaef_ffn_hidden_ratio)
        self.net = nn.Sequential(
            nn.Linear(self.dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.dim),
        )

    def forward(self, x):
        return self.net(x)


class MMAEF_HP(nn.Module):
    """多标签多头注意力增强功能模块 (超参化) - 固定维度512"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dim = 512
        self.num_labels = config.num_classes
        self.num_heads = config.mmaef_num_heads
        self.ffn_hidden_ratio = config.mmaef_ffn_hidden_ratio
        self.use_dropout = config.mmaef_use_dropout
        self.dropout_rate = config.mmaef_dropout_rate
        assert self.dim % self.num_heads == 0, f"dim({self.dim}) 必须能被 num_heads({self.num_heads}) 整除"
        assert self.num_labels > 0, "num_labels 必须大于0"
        assert self.ffn_hidden_ratio > 0, "ffn_hidden_ratio 必须大于0"

        self.self_attention = MultiHeadSelfAttention_HP(config)
        self.cross_attention = MultiHeadCrossAttention_HP(config)
        self.mss_module = MSSModule_HP(config)
        self.ffn = FeedForwardNetwork_HP(config)

        self.norm1 = nn.LayerNorm(self.dim)
        self.norm2 = nn.LayerNorm(self.dim)
        self.norm3 = nn.LayerNorm(self.dim)

        if self.use_dropout:
            self.dropout1 = nn.Dropout(self.dropout_rate)
            self.dropout2 = nn.Dropout(self.dropout_rate)
            self.dropout3 = nn.Dropout(self.dropout_rate)
            self.dropout4 = nn.Dropout(self.dropout_rate)
        else:
            self.dropout1 = self.dropout2 = self.dropout3 = self.dropout4 = nn.Identity()

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, label_emb, image_features):
        B, N_l, C_l = label_emb.shape
        _, _, C_i = image_features.shape
        if C_l != self.dim:
            raise ValueError(f"label_emb的维度({C_l})必须为{self.dim}")
        if C_i != self.dim:
            raise ValueError(f"image_features的维度({C_i})必须为{self.dim}")
        if N_l != self.num_labels:
            print(f"[MMAEF-HP警告] 输入标签数量({N_l})与预期标签数量({self.num_labels})不匹配")

        sa_output = self.self_attention(label_emb)
        norm1_output = self.dropout1(self.norm1(sa_output))
        residual1_output = label_emb + norm1_output

        ca_output = self.cross_attention(residual1_output, image_features)
        norm2_output = self.dropout2(self.norm2(ca_output))
        residual2_output = residual1_output + norm2_output

        mss_output = self.mss_module(residual2_output)
        norm3_output = self.dropout3(self.norm3(mss_output))
        residual3_output = residual2_output + norm3_output

        ffn_output = self.dropout4(self.ffn(residual3_output))
        return residual3_output + ffn_output

