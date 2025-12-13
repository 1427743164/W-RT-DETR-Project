import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def convert_visdrone_to_yolo(visdrone_path):
    """
    将 VisDrone 格式标签转换为 YOLO 格式。
    VisDrone: <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
    YOLO: <class_id> <x_center> <y_center> <width> <height> (全部归一化到 0-1)
    """
    # VisDrone 类别映射 (忽略 0:Ignore, 11:Others)
    # 我们只取 1-10 类，并映射到 0-9
    # 原始: 1:pedestrian, 2:people, 3:bicycle, 4:car, 5:van, 6:truck, 7:tricycle, 8:awning-tricycle, 9:bus, 10:motor
    class_map = {
        1: 0, 2: 1, 3: 2, 4: 3, 5: 4,
        6: 5, 7: 6, 8: 7, 9: 8, 10: 9
    }

    base_path = Path(visdrone_path)

    # 需要处理的文件夹
    splits = ['VisDrone2019-DET-train', 'VisDrone2019-DET-val', 'VisDrone2019-DET-test-dev']

    for split in splits:
        img_dir = base_path / split / 'images'
        label_dir = base_path / split / 'annotations'
        save_dir = base_path / split / 'labels'  # YOLO 需要 labels 文件夹

        if not label_dir.exists():
            print(f"⚠️ 跳过 {split}: 找不到 annotations 文件夹")
            continue

        # 创建 labels 文件夹
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"🚀 正在转换 {split} ...")

        # 遍历标注文件
        for label_file in tqdm(list(label_dir.glob('*.txt'))):
            with open(label_file, 'r') as f:
                lines = f.readlines()

            # 获取对应的图片尺寸
            img_file = img_dir / (label_file.stem + '.jpg')
            if not img_file.exists():
                continue

            try:
                with Image.open(img_file) as img:
                    img_w, img_h = img.size
            except:
                continue

            yolo_lines = []
            for line in lines:
                data = line.strip().split(',')
                if len(data) < 8: continue

                category = int(data[5])

                # 过滤掉 Ignore(0) 和 Others(11)
                if category not in class_map:
                    continue

                cls_id = class_map[category]
                x_min, y_min, w, h = map(float, data[:4])

                # 计算归一化坐标
                x_center = (x_min + w / 2) / img_w
                y_center = (y_min + h / 2) / img_h
                w_norm = w / img_w
                h_norm = h / img_h

                # 边界保护
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                w_norm = max(0, min(1, w_norm))
                h_norm = max(0, min(1, h_norm))

                yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

            # 保存转换后的标签
            with open(save_dir / label_file.name, 'w') as f:
                f.write('\n'.join(yolo_lines))

    print("✅ 转换完成！请修改 data/visdrone.yaml 指向新的 labels 文件夹 (通常 Ultralytics 会自动识别)。")


# === 使用说明 ===
# 请把下面的路径改成你 data/visdrone.yaml 里写的 path 绝对路径
# 例如: D:\pythonProjects\W-RT-DETR-Project\datasets\VisDrone2019-DET
my_visdrone_path = r"D:\pythonProjects\W-RT-DETR-Project\datasets\VisDrone2019-DET"

if __name__ == "__main__":
    convert_visdrone_to_yolo(my_visdrone_path)