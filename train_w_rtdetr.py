from ultralytics import RTDETR
import torch
import sys
import os

def continueTrain():
    # 1. 加载“最后一次存档”
    # 注意：路径一定要对！指向你的 last.pt
    model = RTDETR('./W-RT-DETR-Runs/visdrone_exp_v1/weights/last.pt')

    # 2. 开启续训模式 (resume=True)
    # 不需要再写 data, epochs, batch 等参数了，因为它会从 last.pt 里自动读取之前的配置
    results = model.train(resume=True)


def main():
    # 1. 设置设备
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Training Device: {device}")

    # 2. 强力清理显存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3. 构建模型
    # 先读取你的配置文件，建立 W-RT-DETR 架构
    model = RTDETR('w-rtdetr-l.yaml')

    # 4. 加载预训练权重 (关键步骤)
    # 这会抛出警告说 "Missing keys"（因为你的 backbone 变了），这是完全正常的！
    # 不要因为看到警告就觉得错了，只要 Head 加载进去了就行。
    try:
        if not os.path.exists('rtdetr-l.pt'):
            print("⚠️ 本地未找到 rtdetr-l.pt，正在尝试自动下载...")

        # 加载权重，strict=False 会自动忽略不匹配的小波层
        model = model.load('rtdetr-l.pt')
        print("✅ 成功加载预训练权重 (Head 部分已继承，Backbone 将重新学习)")
    except Exception as e:
        print(f"⚠️ 权重加载跳过: {e}")

    # 5. 开始训练
    results = model.train(
        data='data/visdrone.yaml',
        epochs=100,
        imgsz=640,
        batch=2,
        workers=0,  # Windows 必须为 0

        # === 🟢 显式增强 Warmup (让 NWD 更稳) ===
        warmup_epochs=5,  # 从默认 3 轮增加到 5 轮，给模型更多适应时间
        warmup_bias_lr=0.05,  # 预热时的 Bias 学习率调低一点
        warmup_momentum=0.5,  # 预热时的动量调低，起步更柔和
        # ========================================

        optimizer='AdamW',
        lr0=0.0001,
        project='W-RT-DETR-Runs',
        name='visdrone_pretrained_v1',

        amp=True,  # 混合精度，如果 NWD 报错 NaN 就改成 False
        plots=True,
        exist_ok=True
    )

    print("✅ 训练完成！")


if __name__ == '__main__':
    main()