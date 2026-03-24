#!/usr/bin/env python3
"""
批量处理所有菜品识别训练图片
"""

import subprocess
import sys
from pathlib import Path

# 定义要处理的文件夹
FOLDERS = [
    ('1月15日普遍模糊', '/home/luyicheng/Pictures/菜品识别训练图片/1月15日普遍模糊'),
    ('1月19日早餐', '/home/luyicheng/Pictures/菜品识别训练图片/1月19日/早餐'),
    ('1月19日午餐', '/home/luyicheng/Pictures/菜品识别训练图片/1月19日/午餐'),
    ('1月21日', '/home/luyicheng/Pictures/菜品识别训练图片/1月21日'),
]

def process_folder(folder_name, input_path):
    """处理单个文件夹"""
    output_path = f'/home/luyicheng/project/dish_detection/output_{folder_name}'

    print(f"\n{'='*60}")
    print(f"处理文件夹: {folder_name}")
    print(f"{'='*60}")

    cmd = [
        '/home/luyicheng/yolo_env/bin/python3',
        '/home/luyicheng/project/dish_detection/segment_and_crop_v2.py',
        '--input', input_path,
        '--output', output_path,
        '--confidence', '0.5'
    ]

    try:
        result = subprocess.run(cmd, check=True)
        print(f"✅ {folder_name} 处理完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {folder_name} 处理失败: {e}")
        return False

def main():
    print("=" * 60)
    print("批量处理菜品识别训练图片")
    print("=" * 60)
    print(f"共 {len(FOLDERS)} 个文件夹待处理\n")

    success_count = 0
    fail_count = 0

    for folder_name, input_path in FOLDERS:
        if process_folder(folder_name, input_path):
            success_count += 1
        else:
            fail_count += 1

    print(f"\n{'='*60}")
    print("批量处理完成")
    print(f"{'='*60}")
    print(f"成功: {success_count}/{len(FOLDERS)}")
    print(f"失败: {fail_count}/{len(FOLDERS)}")

if __name__ == '__main__':
    main()