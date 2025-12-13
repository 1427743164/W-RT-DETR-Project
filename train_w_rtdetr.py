from ultralytics import RTDETR  # 也可以用 YOLO 类，但用 RTDETR 更明确
import torch

def continueTrain():
    # 1. 加载“最后一次存档”
    # 注意：路径一定要对！指向你的 last.pt
    model = RTDETR('./W-RT-DETR-Runs/visdrone_exp_v1/weights/last.pt')

    # 2. 开启续训模式 (resume=True)
    # 不需要再写 data, epochs, batch 等参数了，因为它会从 last.pt 里自动读取之前的配置
    results = model.train(resume=True)


def main():
    # # ---------------------------------------------------
    # # 1. 设置设备 (自动检测 GPU)
    # # ---------------------------------------------------
    # device = '0' if torch.cuda.is_available() else 'cpu'
    # print(f"🚀 Training Device: {device}")
    #
    # # ---------------------------------------------------
    # # 2. 加载模型 (构建 W-RT-DETR)
    # # ---------------------------------------------------
    # # 注意：这里加载的是 .yaml 配置文件，表示从头开始构建网络结构
    # # 它会自动读取你的 w-rtdetr-l.yaml，并调用 block.py 里的 FrequencyAwareFusion
    # model = RTDETR('w-rtdetr-l.yaml')
    #
    # # (可选) 如果你想加载预训练权重来加速收敛 (比如官方的 rtdetr-l.pt)
    # # 你可以先加载权重，但由于我们改了网络层数和结构，部分权重可能会由 strict=False 忽略
    # # model = RTDETR('rtdetr-l.pt')
    # # model = RTDETR('w-rtdetr-l.yaml').load('rtdetr-l.pt') # 这种混合写法也可以尝试
    #
    # # ---------------------------------------------------
    # # 3. 开始训练 (Start Training)
    # # ---------------------------------------------------
    # results = model.train(
    #     data='data/visdrone.yaml',  # 数据集配置
    #     epochs=100,  # 训练轮数 (论文建议 72-100)
    #     imgsz=640,  # 输入图像尺寸 (VisDrone 建议 640 或 1024)
    #     batch=2,  # 批次大小 (根据你显存调整，显存大可以设为 8 或 16)
    #
    #     # 优化参数
    #     optimizer='AdamW',  # RT-DETR 标配优化器
    #     lr0=0.0001,  # 初始学习率
    #
    #     # 工程参数
    #     device=device,  # 使用 GPU
    #     project='W-RT-DETR-Runs',  # 训练日志保存的根目录
    #     name='visdrone_exp_v1',  # 本次实验的名称 (结果会存在 W-RT-DETR-Runs/visdrone_exp_v1)
    #     workers=4,  # 数据加载线程数
    #     amp=False,  # 如果遇到 NWD Loss 导致的 NaN 错误，设为 False 关闭混合精度
    #
    #     # 调试参数 (可选)
    #     exist_ok=True,  # 如果目录存在是否覆盖
    #     plots=True  # 自动画出混淆矩阵和训练曲线
    # )


    continueTrain()


    print("✅ 训练完成！Check your results in 'W-RT-DETR-Runs/'")


if __name__ == '__main__':
    # Windows 下的多进程保护
    main()