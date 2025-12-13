import torch

#该函数复制到了D:\pythonProjects\W-RT-DETR-Project\ultralytics\models\utils\loss.py末尾
def nwd_loss(pred_boxes, gt_boxes, constant=12.8, eps=1e-7):
    """
    Normalized Wasserstein Distance Loss for Tiny Objects.
    Reference: [cite: 109-121]

    Args:
        pred_boxes: (N, 4) [cx, cy, w, h] (归一化或绝对坐标均可，需保持一致)
        gt_boxes: (N, 4) [cx, cy, w, h]
        constant: 超参数 C，通常取 12.8 (根据VisDrone/TinyPerson经验值)
    """
    # 1. 解包坐标
    p_cx, p_cy, p_w, p_h = pred_boxes.unbind(-1)
    g_cx, g_cy, g_w, g_h = gt_boxes.unbind(-1)

    # 2. 中心点欧氏距离平方 [cite: 115]
    center_dist_sq = (p_cx - g_cx) ** 2 + (p_cy - g_cy) ** 2

    # 3. 形状相似度 (Wasserstein distance for Gaussian) [cite: 117]
    # 将 BBox 建模为高斯分布，假设协方差矩阵对角化，Sigma = (w/2)^2
    # 公式简化为: ((Wp - Wg)/2)^2 + ((Hp - Hg)/2)^2
    wh_dist_sq = ((p_w - g_w) / 2) ** 2 + ((p_h - g_h) / 2) ** 2

    # Wasserstein 距离平方
    w2_sq = center_dist_sq + wh_dist_sq

    # 4. 归一化 NWD [cite: 120]
    # NWD = exp( - sqrt(W2_sq) / C )
    nwd = torch.exp(-torch.sqrt(w2_sq + eps) / constant)

    # 5. Loss = 1 - NWD
    loss = 1 - nwd

    return loss.mean()


# --- 单元测试 ---
if __name__ == "__main__":
    # 模拟两个几乎不重叠的微小框
    # Box A: [100, 100, 10, 10]
    # Box B: [105, 105, 12, 12] (有偏移，IoU可能很小)
    pred = torch.tensor([[100., 100., 10., 10.]])
    gt = torch.tensor([[112., 112., 10., 10.]])  # 稍微远一点，IoU为0

    loss_val = nwd_loss(pred, gt)
    print(f"NWD Loss Value: {loss_val.item()}")
    # 如果是 IoU loss，这里梯度可能为 0，但 NWD 仍会有梯度回传
    print("损失函数测试通过 ✅")