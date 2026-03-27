#!/usr/bin/env python3
"""
将MOT20数据集的GT标注（gt.txt）转换为YOLO格式，包含track ID信息
这对于JDE模型的训练非常重要，可以学习ReID特征
"""

import os
import glob
import pandas as pd
import shutil
from pathlib import Path

def get_frame_dimensions(seqinfo_file):
    """从seqinfo.ini文件读取帧尺寸"""
    width, height = None, None
    with open(seqinfo_file, 'r') as f:
        for line in f:
            if line.startswith('imWidth'):
                width = int(line.split('=')[1].strip())
            elif line.startswith('imHeight'):
                height = int(line.split('=')[1].strip())
    return width, height

def convert_mot20_gt_to_yolo(mot20_root, output_dir, train_sequences=None, val_sequences=None, test_sequences=None):
    """
    将MOT20的GT标注转换为YOLO格式（包含track ID）
    
    Args:
        mot20_root: MOT20数据集根目录
        output_dir: 输出目录
        train_sequences: 训练序列列表，如['MOT20-01', 'MOT20-02', 'MOT20-05']
        val_sequences: 验证序列列表，如['MOT20-02']
        test_sequences: 测试序列列表，如['MOT20-06', 'MOT20-08']（在test目录下）
    """
    mot20_root = Path(mot20_root)
    output_dir = Path(output_dir)
    
    # 默认序列分配
    if train_sequences is None:
        train_sequences = ['MOT20-01', 'MOT20-02', 'MOT20-05']
    if val_sequences is None:
        val_sequences = ['MOT20-03']
    if test_sequences is None:
        test_sequences = ['MOT20-06', 'MOT20-08']
    
    # 创建输出目录结构
    images_train_dir = output_dir / 'images' / 'train'
    images_val_dir = output_dir / 'images' / 'val'
    images_test_dir = output_dir / 'images' / 'test'
    labels_train_dir = output_dir / 'labels' / 'train'
    labels_val_dir = output_dir / 'labels' / 'val'
    labels_test_dir = output_dir / 'labels' / 'test'
    
    for dir_path in [images_train_dir, images_val_dir, images_test_dir, labels_train_dir, labels_val_dir, labels_test_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # 全局track ID偏移量（确保不同序列的track ID唯一）
    global_track_id_offset = 0
    
    def process_sequence(sequence_name, split_type):
        """处理单个序列"""
        nonlocal global_track_id_offset
        
        # 尝试在train目录下查找
        sequence_path = mot20_root / 'train' / sequence_name
        if not sequence_path.exists():
            # 如果train目录下不存在，尝试test目录
            sequence_path = mot20_root / 'test' / sequence_name
            if not sequence_path.exists():
                print(f"警告: 序列 {sequence_name} 不存在，跳过")
                return
        
        img_folder = sequence_path / 'img1'
        gt_file = sequence_path / 'gt' / 'gt.txt'
        det_file = sequence_path / 'det' / 'det.txt'
        seqinfo_file = sequence_path / 'seqinfo.ini'
        
        # 检查必要文件（测试集可能没有GT文件）
        if not img_folder.exists() or not seqinfo_file.exists():
            print(f"警告: 序列 {sequence_name} 缺少必要文件（img1或seqinfo.ini），跳过")
            return
        
        # 测试集可能没有GT文件，尝试使用det文件
        has_gt = gt_file.exists()
        has_det = det_file.exists()
        if not has_gt and split_type != 'test':
            print(f"警告: 序列 {sequence_name} 缺少GT文件，跳过")
            return
        elif not has_gt and split_type == 'test':
            if has_det:
                print(f"提示: 序列 {sequence_name} 是测试集，使用det.txt作为标注")
            else:
                print(f"提示: 序列 {sequence_name} 是测试集，没有GT和det文件，仅复制图像")
        
        # 获取帧尺寸
        frame_width, frame_height = get_frame_dimensions(seqinfo_file)
        if frame_width is None or frame_height is None:
            print(f"警告: 无法读取序列 {sequence_name} 的帧尺寸，跳过")
            return
        
        # 确定输出目录
        if split_type == 'train':
            images_output_dir = images_train_dir
            labels_output_dir = labels_train_dir
        elif split_type == 'test':
            images_output_dir = images_test_dir
            labels_output_dir = labels_test_dir
        else:  # val
            images_output_dir = images_val_dir
            labels_output_dir = labels_val_dir
        
        # 如果有GT文件，处理标注
        if has_gt:
            annotation_file = gt_file
            use_track_id = True
        elif has_det:
            # 使用det文件（测试集）
            annotation_file = det_file
            use_track_id = False
        else:
            annotation_file = None
        
        if annotation_file:
            # 读取标注文件（GT或det）
            # GT格式: frame_id, track_id, x, y, w, h, conf, class_id, visibility (9列)
            # det格式: frame_id, -1, x, y, w, h, conf, -1, -1, -1 (10列)
            df = pd.read_csv(annotation_file, header=None)
            
            if use_track_id:
                # GT文件格式
                df.columns = ['frame_id', 'track_id', 'x', 'y', 'w', 'h', 'conf', 'class_id', 'vis']
                # 调整track ID以确保全局唯一性
                df['track_id'] += global_track_id_offset
                global_track_id_offset = df['track_id'].max() + 1
                # 过滤有效类别和可见性
                # MOT20中class_id=1是行人
                valid_classes = [1]
                df = df[(df['class_id'].isin(valid_classes)) & (df['vis'] >= 0.1)]
            else:
                # det文件格式（10列）
                df.columns = ['frame_id', 'track_id', 'x', 'y', 'w', 'h', 'conf', 'class_id', 'vis', 'extra']
                # det文件中track_id是-1，保持为-1（表示没有跟踪ID，因为测试集的det.txt不包含跟踪信息）
                df['track_id'] = -1
                # det文件不需要过滤，所有检测框都保留
                # 但可以过滤低置信度的检测框（如果conf列有意义）
                # df = df[df['conf'] > 0.5]  # 可选：过滤低置信度
            
            # 裁剪边界框到图像尺寸内
            df['x'] = df['x'].clip(0, frame_width)
            df['y'] = df['y'].clip(0, frame_height)
            df['w'] = df['w'].clip(0, frame_width - df['x'])
            df['h'] = df['h'].clip(0, frame_height - df['y'])
            
            # 转换为YOLO格式（归一化坐标）
            df['x_center'] = (df['x'] + df['w'] / 2) / frame_width
            df['y_center'] = (df['y'] + df['h'] / 2) / frame_height
            df['bbox_width'] = df['w'] / frame_width
            df['bbox_height'] = df['h'] / frame_height
            
            # 获取所有有标注的帧ID
            annotated_frames = set(df['frame_id'].unique())
            
            # 按帧分组并保存标注
            for frame_id, group in df.groupby('frame_id'):
                frame_annotations = []
                for _, row in group.iterrows():
                    # YOLO格式: class_id x_center y_center width height track_id
                    # class_id=0表示person（在YOLO格式中）
                    yolo_line = (
                        f"0 {row['x_center']:.6f} {row['y_center']:.6f} "
                        f"{row['bbox_width']:.6f} {row['bbox_height']:.6f} "
                        f"{int(row['track_id'])}"
                    )
                    frame_annotations.append(yolo_line)
                
                # 生成输出文件名
                output_filename = f"{sequence_name}_{int(frame_id):06d}"
                
                # 保存标签文件
                label_file = labels_output_dir / f"{output_filename}.txt"
                with open(label_file, 'w') as f:
                    f.write('\n'.join(frame_annotations))
            
            # 复制所有图像文件（包括没有标注的帧，特别是对于测试集）
            img_files = sorted(glob.glob(str(img_folder / "*.jpg")))
            copied_count = 0
            for img_path in img_files:
                img_filename = Path(img_path).name
                try:
                    frame_id = int(img_filename.replace('.jpg', ''))
                    output_filename = f"{sequence_name}_{frame_id:06d}"
                    
                    # 复制图像文件
                    dst_img = images_output_dir / f"{output_filename}.jpg"
                    shutil.copy2(img_path, dst_img)
                    copied_count += 1
                    
                    # 如果该帧没有标注，创建空的标签文件
                    if frame_id not in annotated_frames:
                        label_file = labels_output_dir / f"{output_filename}.txt"
                        label_file.touch()
                except ValueError:
                    continue
            
            print(f"✓ 完成序列 {sequence_name} ({split_type}): {len(df)} 个标注，{copied_count} 帧图像")
        else:
            # 没有GT文件，仅复制图像（用于测试集）
            img_files = sorted(glob.glob(str(img_folder / "*.jpg")))
            frame_count = 0
            for img_path in img_files:
                img_filename = Path(img_path).name
                # 从文件名提取帧号（格式：000001.jpg）
                try:
                    frame_id = int(img_filename.replace('.jpg', ''))
                    output_filename = f"{sequence_name}_{frame_id:06d}"
                    
                    # 复制图像文件
                    dst_img = images_output_dir / f"{output_filename}.jpg"
                    shutil.copy2(img_path, dst_img)
                    
                    # 创建空的标签文件
                    label_file = labels_output_dir / f"{output_filename}.txt"
                    label_file.touch()
                    
                    frame_count += 1
                except ValueError:
                    continue
            
            print(f"✓ 完成序列 {sequence_name} ({split_type}): {frame_count} 帧图像（无标注）")
    
    # 处理训练序列
    print("处理训练序列...")
    for seq in train_sequences:
        process_sequence(seq, 'train')
    
    # 处理验证序列
    if val_sequences:
        print("处理验证序列...")
        for seq in val_sequences:
            process_sequence(seq, 'val')
    
    # 处理测试序列
    if test_sequences:
        print("处理测试序列...")
        for seq in test_sequences:
            process_sequence(seq, 'test')
    
    # 创建配置文件
    config_file = output_dir / 'MOT20_label.yaml'
    val_line = "val: images/val  # val images (relative to 'path')\n" if val_sequences else ""
    with open(config_file, 'w') as f:
        f.write(f"""# MOT20数据集（使用GT标注，包含track ID）
# 此配置用于JDE模型训练，支持ReID学习

path: {output_dir.absolute()}  # dataset root dir
train: images/train  # train images (relative to 'path')
{val_line}test: images/test  # test images (relative to 'path')

# Classes
names:
  0: person
""")
    
    print(f"\n✓ 转换完成！")
    print(f"✓ 输出目录: {output_dir}")
    print(f"✓ 配置文件: {config_file}")
    print(f"\n使用方法:")
    print(f"  data='{config_file}'  # 在train.py中使用此配置")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='将MOT20 GT标注转换为YOLO格式（包含track ID）')
    parser.add_argument('--mot20_root', type=str, default='/home/bns/sevenT/ly/datasets/MOT20',
                        help='MOT20数据集根目录')
    parser.add_argument('--output_dir', type=str, default='/home/bns/sevenT/ly/datasets/MOT20_YOLO_label',
                        help='输出目录')
    parser.add_argument('--train_seqs', nargs='+', default=['MOT20-01', 'MOT20-02', 'MOT20-05'],
                        help='训练序列列表')
    parser.add_argument('--val_seqs', nargs='+', default=['MOT20-03'],
                        help='验证序列列表')
    parser.add_argument('--test_seqs', nargs='+', default=['MOT20-06', 'MOT20-08'],
                        help='测试序列列表')
    
    args = parser.parse_args()
    
    convert_mot20_gt_to_yolo(
        mot20_root=args.mot20_root,
        output_dir=args.output_dir,
        train_sequences=args.train_seqs,
        val_sequences=args.val_seqs,
        test_sequences=args.test_seqs
    )
