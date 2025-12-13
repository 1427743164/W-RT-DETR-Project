import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def haar_dwt_visualization(img_path, save_name='wavelet_analysis.png'):
    # 1. 读取图片
    if not os.path.exists(img_path):
        print(f"❌ 错误：找不到图片 {img_path}")
        return

    img_raw = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

    # 转为 Tensor (1, C, H, W)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    # 2. 手动实现 Haar 小波变换 (模拟你的 DWT 模块)
    # 这样写是为了解耦，不用依赖你项目里的 models 文件
    def get_haar_kernels(device):
        ll = torch.tensor([[1, 1], [1, 1]], device=device).float() / 2.0
        lh = torch.tensor([[-1, -1], [1, 1]], device=device).float() / 2.0
        hl = torch.tensor([[-1, 1], [-1, 1]], device=device).float() / 2.0
        hh = torch.tensor([[1, -1], [-1, 1]], device=device).float() / 2.0
        kernels = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)  # (4, 1, 2, 2)
        return kernels

    # 对 R, G, B 三通道分别做 DWT
    kernels = get_haar_kernels(img_tensor.device)
    # group=3 表示对 RGB 分别卷积
    kernels = torch.cat([kernels] * 3, dim=0)

    # 使用 stride=2 进行下采样
    out = F.conv2d(img_tensor, kernels, stride=2, groups=3)

    # 拆分通道 (B, 12, H, W) -> LL, LH, HL, HH
    # 注意：这里的通道排列取决于 conv2d 的输出顺序，这里简化处理用于可视化
    # 我们把 RGB 的 LL 合并，RGB 的 LH 合并...
    ll = out[:, 0::4, :, :]
    lh = out[:, 1::4, :, :]
    hl = out[:, 2::4, :, :]
    hh = out[:, 3::4, :, :]

    # 3. 计算“高频能量密度” (Energy Density)
    # 公式：Energy = sqrt(LH^2 + HL^2 + HH^2)
    high_freq_energy = torch.sqrt(lh ** 2 + hl ** 2 + hh ** 2)
    # 取 RGB 平均值变成单通道热力图
    energy_map = high_freq_energy.mean(dim=1).squeeze().numpy()

    # 归一化以便显示
    energy_map = (energy_map - energy_map.min()) / (energy_map.max() - energy_map.min())

    # 4. 准备可视化子带 (取平均变成灰度图)
    def to_numpy(x):
        return x.mean(dim=1).squeeze().numpy()  # (H, W)

    titles = ['Original', 'LL (Low Freq)', 'LH (Horizontal)', 'HL (Vertical)', 'HH (Diagonal)', 'Energy Heatmap']

    # 绘图
    plt.figure(figsize=(24, 4))

    # 原图
    plt.subplot(1, 6, 1)
    plt.imshow(img_rgb)
    plt.title(titles[0])
    plt.axis('off')

    # LL
    plt.subplot(1, 6, 2)
    plt.imshow(to_numpy(ll), cmap='gray')
    plt.title(titles[1])
    plt.axis('off')

    # LH
    plt.subplot(1, 6, 3)
    plt.imshow(to_numpy(lh), cmap='gray')
    plt.title(titles[2])
    plt.axis('off')

    # HL
    plt.subplot(1, 6, 4)
    plt.imshow(to_numpy(hl), cmap='gray')
    plt.title(titles[3])
    plt.axis('off')

    # HH
    plt.subplot(1, 6, 5)
    plt.imshow(to_numpy(hh), cmap='gray')
    plt.title(titles[4])
    plt.axis('off')

    # Energy Heatmap (最重要的一张)
    plt.subplot(1, 6, 6)
    plt.imshow(energy_map, cmap='jet')  # 使用 jet 颜色映射，越红能量越高
    plt.title(titles[5])
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    print(f"✅ 原理分析图已保存为: {save_name}")


if __name__ == "__main__":
    # 🔥🔥🔥 记得把这里换成你 VisDrone 数据集里一张车比较多的图片路径 🔥🔥🔥
    test_img = 'datasets/VisDrone2019-DET/VisDrone2019-DET-train/images/0000006_00159_d_0000007.jpg'
    # 如果找不到上面的图，随便换一张存在的 jpg

    haar_dwt_visualization(test_img)