import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarWaveletDownsampling(nn.Module):
    """
    [优化点1: 可学习小波]
    不再使用固定的 Haar 核，而是将其初始化为 Haar 但允许通过反向传播微调。
    这实现了 "Data-Dependent Wavelet"，能自适应 VisDrone 的特殊纹理。
    """

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        # 1. 定义标准 Haar 核作为初始值
        haar_weights = torch.tensor([
            [[[0.5, 0.5], [0.5, 0.5]]],  # LL
            [[[-0.5, -0.5], [0.5, 0.5]]],  # LH
            [[[-0.5, 0.5], [-0.5, 0.5]]],  # HL
            [[[0.5, -0.5], [-0.5, 0.5]]]  # HH
        ], dtype=torch.float32)

        # 2. 关键修改：使用 nn.Parameter 替代 register_buffer
        # requires_grad=True 让网络在训练中可以优化这些参数
        self.weight = nn.Parameter(haar_weights.repeat(in_channels, 1, 1, 1), requires_grad=True)

    def forward(self, x):
        # 使用 Group Conv2d 实现通道独立的变换
        return F.conv2d(x, self.weight, stride=2, groups=self.in_channels)


class FrequencyAwareFusion(nn.Module):
    """
    [优化点2: 跨频带交互融合]
    低频 (LL) 包含语义和结构信息，高频 (High) 包含边缘和噪声。
    我们利用 LL 生成注意力掩码，去“清洗”高频分量，抑制无意义的背景噪声。
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.haar = HaarWaveletDownsampling(in_channels)

        # 频率通道数: 4 * in_channels (LL, LH, HL, HH)
        freq_c = 4 * in_channels

        # 1. 交互模块：从 LL 映射出 Attention Map
        # 输入是 LL (C)，输出是 High (3C) 的权重
        self.cross_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局视野
            nn.Conv2d(in_channels, in_channels * 3, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels * 3, in_channels * 3, 1),
            nn.Sigmoid()  # 生成 0~1 的门控
        )

        # 2. 最终融合
        self.fusion_conv = nn.Conv2d(freq_c, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        # x: (B, C, H, W) -> x_freq: (B, 4C, H/2, W/2)
        x_freq = self.haar(x)

        B, C4, H, W = x_freq.shape
        C = C4 // 4

        # 拆分通道：前 C 个是 LL，后 3C 个是高频 (LH, HL, HH)
        x_ll = x_freq[:, :C, :, :]
        x_high = x_freq[:, C:, :, :]

        # [Cross-Band Interaction]
        # 用 LL 计算出注意力权重，指导高频
        attn_mask = self.cross_attn(x_ll)  # (B, 3C, 1, 1)
        x_high_refined = x_high * attn_mask  # 抑制背景噪声

        # 拼接回去：(LL + Refined_High)
        x_out = torch.cat([x_ll, x_high_refined], dim=1)

        return self.act(self.bn(self.fusion_conv(x_out)))


class WaveletUpsample(nn.Module):
    """
    [优化点3: 频域增强上采样]
    替代粗糙的 nn.Upsample(nearest)。
    使用转置卷积 (Transposed Conv) 学习如何从低分辨率恢复高分辨率细节，
    模拟反小波变换 (IDWT) 的过程。
    """

    def __init__(self, in_channels, scale_factor=2):
        super().__init__()
        # kernel_size=2, stride=2 正好对应 2 倍上采样
        # 初始化时，它可以学成双线性插值，也可以学成 IDWT
        self.up_conv = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.up_conv(x)))