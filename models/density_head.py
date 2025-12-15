import torch
import torch.nn as nn

# 在D:\pythonProjects\W-RT-DETR-Project\ultralytics\nn\modules\head.py导入

class DensityGuidedQuerySelector(nn.Module):
    """
    利用高频能量场和密度预测来初始化 Object Queries。
    Reference: [cite: 36-42], [cite: 126-148]
    """

    def __init__(self, hidden_dim, num_queries=300):
        super().__init__()
        self.num_queries = num_queries

        # 简单的密度预测头 [cite: 130-135]
        self.density_head = nn.Sequential(
            # 第一层 Conv 输出是 hidden_dim // 2 (即 128)
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),

            # [修改这里] GroupNorm 的通道数必须也是 hidden_dim // 2 (即 128)
            nn.GroupNorm(num_groups=32, num_channels=hidden_dim // 2),

            nn.ReLU(),
            # 下一层 Conv 的输入也是 hidden_dim // 2
            nn.Conv2d(hidden_dim // 2, 1, 1),
            nn.Sigmoid()
        )

    def calculate_high_freq_energy(self, haar_feats):
        """
        计算高频能量: E = LH^2 + HL^2 + HH^2
        haar_feats: (B, 4C, H, W) -> 顺序为 LL, LH, HL, HH
        """
        B, C4, H, W = haar_feats.shape
        C = C4 // 4
        # Reshape to separate frequency bands: (B, C, 4, H, W)
        feats_reshaped = haar_feats.view(B, C, 4, H, W)

        # 提取高频分量 (Indices: 1=LH, 2=HL, 3=HH)
        lh = feats_reshaped[:, :, 1, :, :]
        hl = feats_reshaped[:, :, 2, :, :]
        hh = feats_reshaped[:, :, 3, :, :]

        # 计算能量
        energy = (lh.pow(2) + hl.pow(2) + hh.pow(2)).mean(dim=1, keepdim=True)  # 平均各通道能量
        return energy

    def forward(self, feature_map, haar_feats=None):
        """
        Args:
            feature_map: Backbone 输出的特征图 (B, C, H, W)
            haar_feats: (可选) 小波变换后的原始特征，用于辅助能量计算
        """
        # 1. 预测密度图
        density_map = self.density_head(feature_map)  # (B, 1, H, W)

        # 如果提供了小波特征，可以将能量场叠加到密度图中作为先验 (Enhancement)
        if haar_feats is not None:
            energy_field = self.calculate_high_freq_energy(haar_feats)
            # 简单的融合策略：密度图 * (1 + 能量)
            density_map = density_map * (1.0 + energy_field)

        # 2. 展平以进行 Top-K 选择 [cite: 141]
        B, _, H, W = density_map.shape
        density_flat = density_map.flatten(2)  # (B, 1, H*W)

        # 3. 选择密度最高的点作为 Query 位置 [cite: 143]
        topk_scores, topk_indices = torch.topk(density_flat, self.num_queries, dim=2)

        # 4. 转换回坐标 (cx, cy) 归一化坐标 [cite: 145-146]
        topk_y = (topk_indices // W).float() / H
        topk_x = (topk_indices % W).float() / W

        # (B, Num_Queries, 2) -> (cx, cy)
        ref_points = torch.stack([topk_x, topk_y], dim=-1).squeeze(1)

        if self.training and torch.rand(1).item() < 0.01:  # 只有 1% 的概率打印，防止刷屏
            print(
                f"\n🔍 [Density Head Check] Mean: {density_map.mean().item():.4f}, Max: {density_map.max().item():.4f}, Min: {density_map.min().item():.4f}")

        return ref_points, density_map


# --- 单元测试 ---
if __name__ == "__main__":
    dummy_feat = torch.rand(1, 256, 32, 32)
    # 模拟 4倍通道的小波特征 (64*4 = 256)
    dummy_haar = torch.rand(1, 256, 32, 32)

    selector = DensityGuidedQuerySelector(hidden_dim=256, num_queries=300)
    ref_pts, d_map = selector(dummy_feat, dummy_haar)

    print(f"Ref Points Shape: {ref_pts.shape}")  # (1, 300, 2)
    print(f"Density Map Shape: {d_map.shape}")
    print("查询选择模块测试通过 ✅")