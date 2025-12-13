# 新建 predict_sahi.py
from ultralytics import RTDETR
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import cv2
import os


def main():
    # 1. 准备路径
    weight_path = 'W-RT-DETR-Runs/visdrone_exp_v1/weights/best.pt'
    image_path = 'datasets/VisDrone2019-DET/VisDrone2019-DET-test-dev/images/0000006_00159_d_0000007.jpg'

    # 2. 包装你的 W-RT-DETR 模型给 SAHI 使用
    # SAHI 默认支持 YOLO，我们需要用这种方式让它支持 RT-DETR
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',  # 借用接口
        model_path=weight_path,
        confidence_threshold=0.25,
        device="cuda:0",  # 或 'cpu'
    )

    # 3. 核心：切片推理
    # slice_height/width: 切片大小，建议和训练时的 imgsz 一致 (640)
    # overlap_height_ratio: 重叠率，防止切断物体
    print("🚀 开始切片推理 (这可能比普通推理慢，但更精准)...")
    result = get_sliced_prediction(
        image_path,
        detection_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )

    # 4. 保存结果
    save_path = "sahi_result.jpg"
    result.export_visuals(export_dir=".", file_name="sahi_result")
    print(f"✅ SAHI 推理完成！结果图已保存为: {save_path}")


if __name__ == '__main__':
    main()