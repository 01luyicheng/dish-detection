#!/usr/bin/env python3
"""
菜品实例分割处理工具（支持并行处理）
"""

import argparse
import multiprocessing
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
from ultralytics import YOLO
import sys


class DishSegmenter:
    def __init__(self, confidence=0.5):
        self.confidence = confidence
        print("正在加载预训练YOLOv8n-seg模型...")
        self.model = YOLO('yolov8n-seg.pt')
        self.class_names = self.model.names
        self.colors = self._generate_colors()
        print(f"模型加载完成，共{len(self.class_names)}个类别")

    def _generate_colors(self):
        """为每个类别生成不同的颜色"""
        num_classes = len(self.class_names)
        colors = {}
        for i in range(num_classes):
            hue = int((i * 137.508) % 180)
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            colors[i] = tuple(map(int, color))
        return colors

    def process_image(self, image_path, output_dir):
        """处理单张图片"""
        try:
            # 读取图片
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"警告: 无法读取图片 {image_path}")
                return 0

            # 运行推理
            results = self.model(image, conf=self.confidence, device='cpu')

            if len(results) == 0 or len(results[0].boxes) == 0:
                print(f"未检测到物体: {image_path.name}")
                return 0

            # 创建输出目录
            cropped_dir = output_dir / 'cropped_objects'
            annotated_dir = output_dir / 'annotated_images'
            cropped_dir.mkdir(parents=True, exist_ok=True)
            annotated_dir.mkdir(parents=True, exist_ok=True)

            # 复制原图用于标注
            annotated_image = image.copy()

            # 处理每个检测到的物体
            for i, result in enumerate(results[0].boxes):
                box = result.xyxy[0].cpu().numpy()
                conf = result.conf[0].cpu().numpy()
                class_id = int(result.cls[0].cpu().numpy())
                class_name = self.class_names[class_id]

                # 获取掩码
                if results[0].masks is not None:
                    mask = results[0].masks.data[i].cpu().numpy()
                    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
                    mask = (mask > 0.5).astype(np.uint8) * 255
                else:
                    mask = None

                x1, y1, x2, y2 = map(int, box)

                # 绘制标注
                color = self.colors[class_id]

                # 绘制实例分割边缘
                if mask is not None:
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(annotated_image, contours, -1, color, 3)

                # 绘制矩形框
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 3)

                # 绘制标签
                label = f"{class_name} {conf:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
                label_x = x1
                label_y = y1 - 10 if y1 > 30 else y1 + label_size[1] + 10

                # 标签背景
                cv2.rectangle(annotated_image,
                            (label_x, label_y - label_size[1] - 10),
                            (label_x + label_size[0] + 20, label_y + 10),
                            color, -1)

                # 标签文字
                cv2.putText(annotated_image, label,
                           (label_x + 10, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

                # 裁剪物体
                if mask is not None:
                    cropped = cv2.bitwise_and(image, image, mask=mask)
                    cropped = cropped[y1:y2, x1:x2]
                else:
                    cropped = image[y1:y2, x1:x2]

                # 保存裁剪的物体
                filename = f"{class_name}_{conf:.2f}_{image_path.stem}_{i:02d}.jpg"
                cv2.imwrite(str(cropped_dir / filename), cropped)

            # 保存标注的原图
            annotated_filename = f"{image_path.stem}_annotated.jpg"
            cv2.imwrite(str(annotated_dir / annotated_filename), annotated_image)

            return len(results[0].boxes)

        except Exception as e:
            print(f"处理图片 {image_path} 时出错: {e}")
            return 0


def worker_process(args):
    """工作进程函数"""
    image_path, output_dir, confidence = args
    segmenter = DishSegmenter(confidence=confidence)
    count = segmenter.process_image(image_path, output_dir)
    return str(image_path), count


def main():
    parser = argparse.ArgumentParser(description='菜品实例分割处理工具（支持并行处理）')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='输入路径（图片文件或文件夹）')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='输出路径（默认: /home/luyicheng/project/dish_detection/output/annotated_images/日期）')
    parser.add_argument('--confidence', '-c', type=float, default=0.5,
                       help='检测置信度阈值 (默认: 0.5)')
    parser.add_argument('--workers', '-w', type=int, default=None,
                       help='并行工作进程数（默认: CPU核心数）')

    args = parser.parse_args()

    # 获取输入路径
    input_path = Path(args.input)

    # 收集所有图片
    image_files = []
    if input_path.is_file():
        if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            image_files = [input_path]
    elif input_path.is_dir():
        image_files = sorted(list(input_path.glob('*.jpg')) +
                            list(input_path.glob('*.JPG')) +
                            list(input_path.glob('*.png')))

    if not image_files:
        print(f"错误: 在 {input_path} 中没有找到图片")
        sys.exit(1)

    print(f"找到 {len(image_files)} 张图片")

    # 确定输出路径
    if args.output:
        output_dir = Path(args.output)
    else:
        # 使用默认输出路径
        default_output = Path('/home/luyicheng/project/dish_detection/output/annotated_images')
        # 尝试从输入路径提取日期
        date_str = None
        for part in input_path.parts:
            if '月' in part and '日' in part:
                date_str = part
                break
        if date_str:
            output_dir = default_output / date_str
        else:
            # 使用当前日期
            today = datetime.now()
            date_str = f"{today.month}月{today.day}日"
            output_dir = default_output / date_str

    print(f"输出目录: {output_dir}")

    # 确定并行工作进程数
    if args.workers is None:
        workers = multiprocessing.cpu_count()
    else:
        workers = args.workers
    print(f"使用 {workers} 个并行工作进程")

    # 准备任务
    tasks = [(img, output_dir, args.confidence) for img in image_files]

    # 使用进程池并行处理
    print("开始处理...")
    with multiprocessing.Pool(processes=workers) as pool:
        results = pool.map(worker_process, tasks)

    # 统计结果
    total_objects = sum(count for _, count in results)
    processed_count = sum(1 for _, count in results if count > 0)

    print(f"\n处理完成!")
    print(f"处理图片: {processed_count}/{len(image_files)}")
    print(f"检测物体总数: {total_objects}")
    print(f"输出目录: {output_dir}")


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    main()