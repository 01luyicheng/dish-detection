#!/usr/bin/env python3
"""
使用预训练YOLOv8实例分割模型检测并裁剪菜品（增强版）
- 抠出实例
- 在原图上绘制边缘、矩形框和标签
- 文件名包含识别结果和置信度
"""

from ultralytics import YOLO
import cv2
import os
import numpy as np
from pathlib import Path
import shutil

class DishSegmenterV2:
    def __init__(self, image_dir, output_dir, confidence=0.5, imgsz=640):
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.confidence = confidence
        self.imgsz = imgsz  # 图片尺寸（使用640保证准确率）

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cropped_dir = self.output_dir / 'cropped_objects'
        self.annotated_dir = self.output_dir / 'annotated_images'
        self.cropped_dir.mkdir(parents=True, exist_ok=True)
        self.annotated_dir.mkdir(parents=True, exist_ok=True)

        # 获取所有图片
        self.image_files = sorted(list(self.image_dir.glob('*.jpg')) +
                                 list(self.image_dir.glob('*.JPG')) +
                                 list(self.image_dir.glob('*.png')))

        print(f"找到 {len(self.image_files)} 张图片")

        # 加载预训练模型
        print("正在加载预训练YOLOv8n-seg模型...")
        self.model = YOLO('yolov8n-seg.pt')

        # 获取模型类别
        self.class_names = self.model.names
        print(f"模型类别: {self.class_names}")

        # 颜色映射（为每个类别分配不同颜色）
        self.colors = self._generate_colors()

    def _generate_colors(self):
        """为每个类别生成不同的颜色"""
        colors = {}
        for class_id in range(len(self.class_names)):
            # 使用HSV生成鲜艳的颜色
            hue = int((class_id * 137.508) % 180)  # 黄金角度，限制在0-179（OpenCV的HSV范围）
            color = cv2.cvtColor(np.array([[[hue, 255, 255]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)
            colors[class_id] = tuple(map(int, color[0][0]))
        return colors

    def process_images(self):
        """处理所有图片"""
        total_objects = 0
        skipped = 0

        for idx, image_path in enumerate(self.image_files, 1):
            print(f"\n处理 [{idx}/{len(self.image_files)}]: {image_path.name}")

            # 读取图片
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"  跳过: 无法读取图片")
                skipped += 1
                continue

            # 运行推理（CPU，使用较小的图片尺寸以提高速度）
            results = self.model(image, conf=self.confidence, device='cpu', imgsz=self.imgsz)

            # 获取检测结果
            result = results[0]

            if len(result) == 0:
                print(f"  未检测到物体")
                continue

            # 处理每个检测到的物体
            objects_count = self._process_detections(image, result, image_path.stem)
            total_objects += objects_count

            print(f"  检测到 {objects_count} 个物体")

        print(f"\n{'='*60}")
        print(f"处理完成！")
        print(f"总图片数: {len(self.image_files)}")
        print(f"跳过图片: {skipped}")
        print(f"检测物体总数: {total_objects}")
        print(f"裁剪图片保存在: {self.cropped_dir}")
        print(f"标注图片保存在: {self.annotated_dir}")
        print(f"{'='*60}")

    def _process_detections(self, image, result, image_name):
        """处理检测到的物体"""
        count = 0

        # 复制原图用于绘制
        annotated_image = image.copy()

        # 获取分割掩码和边界框
        if result.masks is None:
            return 0

        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        for i, (mask, box, conf, class_id) in enumerate(zip(masks, boxes, confidences, class_ids)):
            # 调整掩码大小到原图尺寸
            h, w = image.shape[:2]
            mask_resized = cv2.resize(mask, (w, h))

            # 创建掩码图像（二值化）
            mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255

            # 获取类别信息
            class_name = self.class_names[class_id]
            color = self.colors[class_id]

            # 1. 在标注图上绘制分割边缘
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(annotated_image, contours, -1, color, 2)

            # 2. 绘制矩形框
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)

            # 3. 绘制标签（类别 + 置信度）
            label = f"{class_name} {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)

            # 确保标签在图片范围内
            label_x = max(0, x1)
            label_y = max(label_size[1] + 10, y1 - 10)

            # 绘制标签背景
            cv2.rectangle(annotated_image,
                         (label_x, label_y - label_size[1] - 10),
                         (label_x + label_size[0] + 20, label_y + 10),
                         color, -1)

            # 绘制标签文字
            cv2.putText(annotated_image, label,
                       (label_x + 10, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

            # 4. 应用掩码到原图并裁剪
            masked_image = cv2.bitwise_and(image, image, mask=mask_binary)

            # 裁剪物体
            cropped = masked_image[y1:y2, x1:x2]

            # 检查裁剪区域是否有效
            if cropped.size == 0:
                continue

            # 创建输出文件名（类别和置信度在最前面）
            filename = f"{class_name}_{conf:.2f}_{image_name}_{i:02d}.jpg"
            output_path = self.cropped_dir / filename

            # 保存裁剪的图片
            cv2.imwrite(str(output_path), cropped)
            count += 1

            print(f"    - {class_name} (置信度: {conf:.2f}) -> {filename}")

        # 保存标注后的原图
        annotated_filename = f"{image_name}_annotated.jpg"
        annotated_path = self.annotated_dir / annotated_filename
        cv2.imwrite(str(annotated_path), annotated_image)

        return count

def main():
    import argparse

    parser = argparse.ArgumentParser(description='使用YOLOv8实例分割检测并裁剪菜品（增强版）')
    parser.add_argument('--input', '-i',
                       default='/home/luyicheng/Pictures/菜品识别训练图片/1月19日/中餐',
                       help='输入图片目录')
    parser.add_argument('--output', '-o',
                       default='/home/luyicheng/project/dish_detection/output_中餐',
                       help='输出目录')
    parser.add_argument('--confidence', '-c', type=float, default=0.5,
                       help='检测置信度阈值 (默认: 0.5)')
    parser.add_argument('--imgsz', '-s', type=int, default=640,
                       help='推理图片尺寸 (默认: 640，保证准确率)')

    args = parser.parse_args()

    print("=" * 60)
    print("菜品实例分割和裁剪工具（增强版）")
    print("=" * 60)
    print(f"输入目录: {args.input}")
    print(f"输出目录: {args.output}")
    print(f"置信度阈值: {args.confidence}")
    print(f"推理图片尺寸: {args.imgsz} (越小越快但精度降低)")
    print("=" * 60)

    # 创建分割器
    segmenter = DishSegmenterV2(
        image_dir=args.input,
        output_dir=args.output,
        confidence=args.confidence,
        imgsz=args.imgsz
    )

    # 处理图片
    segmenter.process_images()

if __name__ == '__main__':
    main()