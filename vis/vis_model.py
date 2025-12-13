from ultralytics import RTDETR
import os
import shutil


def visualize_model_features():
    # 1. 加载你训练好的模型 (Best)
    # 请确保这个路径是对的
    model_path = 'W-RT-DETR-Runs/visdrone_exp_v1/weights/best.pt'

    if not os.path.exists(model_path):
        # 如果还没跑完，先用 last.pt 凑合测试一下代码
        print("⚠️ 没找到 best.pt，尝试寻找 last.pt...")
        model_path = 'W-RT-DETR-Runs/visdrone_exp_v1/weights/last.pt'

    model = RTDETR(model_path)

    # 2. 指定一张测试图片
    # 最好找那种有一大群蚂蚁大小的人或车的图，效果最震撼
    img_path = 'datasets/VisDrone2019-DET/VisDrone2019-DET-val/images/0000006_00159_d_0000007.jpg'  # 示例路径

    # 3. 运行预测并开启可视化
    # visualize=True 是关键
    # project 和 name 指定保存路径
    print("🚀 开始生成特征图，这可能需要几秒钟...")
    results = model.predict(
        source=img_path,
        visualize=True,  # 👈 核心参数：开启特征可视化
        imgsz=640,
        project='runs/visualize',
        name='exp',
        exist_ok=True
    )

    print(f"✅ 可视化完成！")
    print(f"请打开文件夹查看结果: runs/visualize/exp/")
    print("👉 重点找以 'stage' 开头的图片，比如 stage0_... 到 stage3_...")
    print("👉 尤其是 stage2 或 stage3 的 FrequencyAwareFusion 之后的图，应该能看到很多亮点。")


if __name__ == "__main__":
    visualize_model_features()