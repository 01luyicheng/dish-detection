# 菜品实例分割处理工具使用说明

## 功能特点

- ✅ 支持4K分辨率原图处理
- ✅ 多进程并行处理，充分利用CPU
- ✅ 自动生成裁剪的实例图片
- ✅ 自动生成标注的原图（包含实例分割边缘、矩形框、标签）
- ✅ 灵活的输入输出路径配置

## 测试结果

### 单张4K图片处理性能
- **图片分辨率**: 4000x2256
- **处理时间**: 约0.86秒（不含模型加载）
- **内存使用**: 约420MB
- **检测物体**: 2个

### 并行处理性能
- 使用4个并行进程处理121张图片
- 总处理时间: 约5-6分钟
- 平均每张图片: 约3秒

## 使用方法

### 基本用法

```bash
# 处理单个图片文件
/home/luyicheng/yolo_env/bin/python3 process_dishes.py --input /path/to/image.jpg

# 处理整个文件夹
/home/luyicheng/yolo_env/bin/python3 process_dishes.py --input /path/to/folder
```

### 指定输出路径

```bash
# 自定义输出路径
/home/luyicheng/yolo_env/bin/python3 process_dishes.py \
  --input /home/luyicheng/Pictures/菜品识别训练图片/1月19日/午餐 \
  --output /home/luyicheng/project/dish_detection/output/annotated_images/1月19日午餐
```

### 默认输出路径

如果不指定 `--output`，程序会自动从输入路径提取日期，并输出到：
```
/home/luyicheng/project/dish_detection/output/annotated_images/{日期}
```

例如：
- 输入: `/home/luyicheng/Pictures/菜品识别训练图片/1月21日/早餐`
- 输出: `/home/luyicheng/project/dish_detection/output/annotated_images/1月21日`

### 调整并行进程数

```bash
# 使用2个并行进程（适合低内存机器）
/home/luyicheng/yolo_env/bin/python3 process_dishes.py \
  --input /path/to/folder \
  --workers 2

# 使用8个并行进程（适合高性能机器）
/home/luyicheng/yolo_env/bin/python3 process_dishes.py \
  --input /path/to/folder \
  --workers 8
```

**注意**: 每个进程会占用约420MB内存，请根据机器内存调整进程数。

### 调整置信度阈值

```bash
# 提高置信度阈值（减少误检）
/home/luyicheng/yolo_env/bin/python3 process_dishes.py \
  --input /path/to/folder \
  --confidence 0.7

# 降低置信度阈值（检测更多物体）
/home/luyicheng/yolo_env/bin/python3 process_dishes.py \
  --input /path/to/folder \
  --confidence 0.3
```

## 命令行参数

| 参数 | 简写 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--input` | `-i` | string | 必需 | 输入路径（图片文件或文件夹） |
| `--output` | `-o` | string | 自动生成 | 输出路径 |
| `--confidence` | `-c` | float | 0.5 | 检测置信度阈值 |
| `--workers` | `-w` | int | CPU核心数 | 并行工作进程数 |

## 输出结构

```
输出目录/
├── cropped_objects/          # 裁剪的实例图片
│   ├── bowl_0.86_原图名_00.jpg
│   ├── person_0.92_原图名_01.jpg
│   └── ...
└── annotated_images/         # 标注的原图
    ├── 原图名_annotated.jpg
    ├── 原图名_annotated.jpg
    └── ...
```

## 文件名格式

### 裁剪图片
格式: `{类别}_{置信度}_{原图名}_{序号}.jpg`
- 示例: `bowl_0.86_DJI_20260119114108_0999_D_00.jpg`

### 标注图片
格式: `{原图名}_annotated.jpg`
- 示例: `DJI_20260119114108_0999_D_annotated.jpg`

## 标注效果

标注图片包含：
- 🔵 实例分割边缘（彩色轮廓）
- 🔲 矩形框（边界框）
- 📝 标签文字（类别 + 置信度，大字号）

## 常见问题

### Q: 处理速度慢怎么办？
A: 可以尝试：
1. 增加 `--workers` 参数的值（但要注意内存占用）
2. 降低 `--confidence` 参数的值（但会增加误检）

### Q: 内存不足怎么办？
A: 减少 `--workers` 参数的值，建议使用2个进程。

### Q: 如何只处理特定类型的图片？
A: 目前程序会处理所有 `.jpg`、`.JPG`、`.png` 格式的图片。

### Q: 输出路径的日期是从哪里来的？
A: 程序会自动从输入路径中提取包含"月"和"日"的文件夹名作为日期。如果没有找到，则使用当前日期。

## 硬件建议

- **最低配置**: 双核CPU, 2GB内存
- **推荐配置**: 4核CPU, 4GB内存
- **高性能配置**: 8核CPU, 8GB内存

每个进程占用约420MB内存，请根据内存大小调整进程数。