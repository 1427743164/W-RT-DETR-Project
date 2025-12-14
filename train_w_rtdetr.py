from ultralytics import RTDETR
import torch
import sys
import os


def train_restart_with_weights():
    # 1. 设置路径
    # 指向你刚才中断的那个权重文件
    old_weights = r'D:\pythonProjects\W-RT-DETR-Project\W-RT-DETR-Runs\visdrone_pretrained_v1\weights\last.pt'

    print(f"♻️ 正在读取旧权重并迁移到新架构: {old_weights}")

    # 2. 重新构建模型 (关键：从 YAML 构建，保证结构符合当前代码)
    model = RTDETR('w-rtdetr-l.yaml')

    # 3. 强制加载权重
    # 这一步会把 last.pt 里能用的权重都塞进去，尺寸不对的（比如那个变小的 Wavelet层）会自动舍弃或适配
    try:
        model.load(old_weights)
        print("✅ 权重迁移成功！(部分不匹配层已被自动处理)")
    except Exception as e:
        print(f"⚠️ 权重加载警告 (正常现象): {e}")

    # 4. 开始新一轮训练 (注意：不要写 resume=True)
    # 这将创建一个新的实验文件夹，例如 visdrone_pretrained_v2
    results = model.train(
        data='data/visdrone.yaml',
        epochs=70,  # 你之前跑了30轮，这里可以设为剩余的 70 轮，或者直接 100 重新跑
        imgsz=640,
        batch=2,
        workers=4,  # ✅ 确保这里是 4，加速训练！

        # 优化器设置
        optimizer='AdamW',
        lr0=0.0001,

        project='W-RT-DETR-Runs',
        name='visdrone_restarted',  # 改个名字区分

        amp=True,
        plots=True,
        exist_ok=True
    )

def continueTrain():
    # 1. 路径修正：根据你的日志，文件夹名是 visdrone_pretrained_v1
    # 请再次确认你的文件夹里确实有 last.pt
    checkpoint_path = r'D:\pythonProjects\W-RT-DETR-Project\W-RT-DETR-Runs\visdrone_pretrained_v1\weights\last.pt'

    print(f"🔄 正在加载中断的存档: {checkpoint_path}")

    # 2. 加载模型
    model = RTDETR(checkpoint_path)

    # 3. 开启续训
    # 虽然这里写了 workers=4，但上面提到的 args.yaml 修改才是真正的双保险
    results = model.train(
        resume=True,
        workers=4  # 尝试强制覆盖，配合 args.yaml 修改效果最佳
    )


def main():
    # # 1. 设置设备
    # device = '0' if torch.cuda.is_available() else 'cpu'
    # print(f"🚀 Training Device: {device}")
    #
    # # 2. 强力清理显存
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()
    #
    # # 3. 构建模型
    # # 先读取你的配置文件，建立 W-RT-DETR 架构
    # model = RTDETR('w-rtdetr-l.yaml')
    #
    # # 4. 加载预训练权重 (关键步骤)
    # # 这会抛出警告说 "Missing keys"（因为你的 backbone 变了），这是完全正常的！
    # # 不要因为看到警告就觉得错了，只要 Head 加载进去了就行。
    # try:
    #     if not os.path.exists('rtdetr-l.pt'):
    #         print("⚠️ 本地未找到 rtdetr-l.pt，正在尝试自动下载...")
    #
    #     # 加载权重，strict=False 会自动忽略不匹配的小波层
    #     model = model.load('rtdetr-l.pt')
    #     print("✅ 成功加载预训练权重 (Head 部分已继承，Backbone 将重新学习)")
    # except Exception as e:
    #     print(f"⚠️ 权重加载跳过: {e}")
    #
    # # 5. 开始训练
    # results = model.train(
    #     data='data/visdrone.yaml',
    #     epochs=100,
    #     imgsz=640,
    #     batch=2,
    #     workers=4,  # Windows 必须为 0
    #
    #     # === 🟢 显式增强 Warmup (让 NWD 更稳) ===
    #     warmup_epochs=5,  # 从默认 3 轮增加到 5 轮，给模型更多适应时间
    #     warmup_bias_lr=0.05,  # 预热时的 Bias 学习率调低一点
    #     warmup_momentum=0.5,  # 预热时的动量调低，起步更柔和
    #     # ========================================
    #
    #     optimizer='AdamW',
    #     lr0=0.0001,
    #     project='W-RT-DETR-Runs',
    #     name='visdrone_pretrained_v1',
    #
    #     amp=True,  # 混合精度，如果 NWD 报错 NaN 就改成 False
    #     plots=True,
    #     exist_ok=True
    # )

    # continueTrain()

    train_restart_with_weights()

    print("✅ 训练完成！")


if __name__ == '__main__':
    main()