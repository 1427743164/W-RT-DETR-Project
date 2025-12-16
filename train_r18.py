import warnings

warnings.filterwarnings('ignore')  # 屏蔽警告
from ultralytics import RTDETR
import torch
import os


def main():
    print("🚀 启动 VisDrone R18 训练任务...")

    # 1. 自动清理显存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. 准备权重文件 (核心步骤)
    # 我们先实例化一个官方模型，这会触发 ultralytics 自动去官网下载 rtdetr-r18.pt
    # 如果本地没有，它会自动下载，不用你操心
    pt_file = 'rtdetr-r18.pt'
    if not os.path.exists(pt_file):
        print(f"📥 本地未找到 {pt_file}，正在触发自动下载...")
        # 这一步只是为了下载权重，下载完我们不用这个 model 实例
        temp_model = RTDETR('rtdetr-r18.pt')
        del temp_model
        print("✅ 权重下载完成！")

    # 3. 构建你的自定义模型
    # 使用你刚才新建的配置文件
    model = RTDETR('w-rtdetr-r18.yaml')

    # 4. 加载权重
    # 这一步把刚才下载的 R18 权重塞进你的模型
    # 因为你改了 Neck (小波上采样)，所以 Neck 部分的权重会不匹配，报 Warning 是正常的
    # 但 Backbone (ResNet) 会完美加载，这就够了！
    try:
        model.load(pt_file)
        print(f"✅ 成功从 {pt_file} 迁移权重 (Neck部分差异已自动忽略)")
    except Exception as e:
        print(f"ℹ️ 权重加载提示: {e}")

    # 5. 开始训练
    # 既然换了 R18，我们就可以把 Batch Size 拉大，这是提分的关键！
    results = model.train(
        data='data/visdrone.yaml',

        epochs=200,  # 小模型学得快，多跑点（200轮）能出奇迹
        batch=16,  # 🔥 重点：Batch 16！如果爆显存就改成 12 或 8
        imgsz=640,  # 图像大小

        optimizer='AdamW',
        lr0=0.0004,  # 针对 R18 微调了学习率
        warmup_epochs=5,  # 预热 5 轮

        # 增强策略 (VisDrone 三件套)
        mosaic=1.0,  # 开启马赛克
        mixup=0.15,  # 开启混合增强
        copy_paste=0.3,  # 复制粘贴增强（对小目标很有用）

        project='W-RT-DETR-Runs',
        name='visdrone_r18_v1',  # 实验名称
        exist_ok=True,
        amp=True,
        plots=True
    )

    print("🎉 训练结束！")


if __name__ == '__main__':
    main()