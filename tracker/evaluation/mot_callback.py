import os
import cv2
import yaml
import time
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from types import SimpleNamespace

from ultralytics import YOLO
from tracker.evaluation.evaluate import trackeval_evaluation
from ultralytics.trackers import BYTETracker, BOTSORT, SMILEtrack, BoostTrack, JDETracker, YOLOJDETracker

TRACKER_MAP = {"bytetrack": BYTETracker, "botsort": BOTSORT, "smiletrack": SMILEtrack, "boosttrack": BoostTrack, "jdetracker": JDETracker, "yolojdetracker": YOLOJDETracker}

# TODO: CALLBACKS DO NOT WORK WHEN USING DDP
def mot_eval(validator, period=1):
    is_train = validator.training   # Check if the model is being trained
    if is_train:
        if validator.epoch % period != 0 or validator.epoch == 1:
            return  # Evaluate only every 'period' epochs after the first epoch

    # TODO: DEFINE DATASET, externalize
    # 检测数据集类型：如果数据路径包含 MOT20，使用 MOT20，否则使用 MOT17
    data_path = getattr(validator.args, 'data', '')
    if 'MOT20' in str(data_path) or 'mot20' in str(data_path).lower():
        # 使用 MOT20 数据集
        dataset_name = 'MOT20/train'  # MOT20 验证集在 train 目录中（MOT20-02, MOT20-03, MOT20-05）
        # 使用原始 MOT20 数据集路径
        mot20_root = '/root/autodl-tmp/MOT20'
        dataset_root = os.path.join(mot20_root, 'train')
        # MOT20 验证序列：MOT20-02, MOT20-03, MOT20-05
        # 检查序列是否存在
        available_seqs = []
        for seq in ['MOT20-02', 'MOT20-03', 'MOT20-05']:
            seq_path = os.path.join(dataset_root, seq)
            if os.path.exists(seq_path) and os.path.exists(os.path.join(seq_path, 'img1')):
                available_seqs.append(seq)
        seq_names = available_seqs
        if len(seq_names) == 0:
            print("WARNING: No MOT20 validation sequences found. Skipping MOT evaluation.")
            return
        # 创建seqmap文件
        seqmap_dir = os.path.join(dataset_root, 'seqmaps')
        os.makedirs(seqmap_dir, exist_ok=True)
        seqmap_file = os.path.join(seqmap_dir, 'MOT20-train.txt')
        # 如果seqmap文件不存在，创建它
        if not os.path.exists(seqmap_file):
            with open(seqmap_file, 'w') as f:
                f.write('name\n')  # 标题行
                for seq in seq_names:
                    f.write(f'{seq}\n')
            print(f"已创建seqmap文件: {seqmap_file}")
    else:
        # 默认使用 MOT17
        dataset_name = 'MOT17/val_half'
        seqmap_file = './tracker/evaluation/TrackEval/data/gt/mot_challenge/seqmaps/MOT17-val_half.txt'
        dataset_root = os.path.join('./tracker/evaluation/TrackEval/data/gt/mot_challenge/', dataset_name)
        if os.path.exists(dataset_root):
            seq_names = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]
        else:
            print(f"WARNING: MOT17 dataset root not found at {dataset_root}. Skipping MOT evaluation.")
            return

    # Define output folder
    output_folder = os.path.join(str(validator.save_dir), dataset_name, 'data')
    os.makedirs(output_folder, exist_ok=True)

    # Initialize model
    model = YOLO(validator.best if is_train else validator.model_path, task=validator.args.task)

    # Initialize counters
    total_frames = 0
    total_time = 0.0

    # Iterate over sequences
    for seq_name in tqdm(seq_names, desc="Tracking sequences"):
        # Sort images
        img_dir = os.path.join(dataset_root, seq_name, 'img1')
        all_imgs = sorted(os.listdir(img_dir))
        # 过滤出有效的图像文件
        imgs = [img for img in all_imgs if img.endswith('.jpg') or img.endswith('.png')]
        print(f"\n处理序列 {seq_name}: 共 {len(imgs)} 张图像")

        # Initialize here to restart the tracker for each sequence
        tracker_name = validator.args.tracker.split('.')[0]
        # 获取项目根目录（向上3级目录从 tracker/evaluation/ 到项目根目录）
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        tracker_cfg_path = os.path.join(project_root, 'ultralytics', 'cfg', 'trackers', f'{tracker_name}.yaml')
        tracker_cfg = dict_to_namespace(yaml.safe_load(open(tracker_cfg_path)))
        tracker = TRACKER_MAP[tracker_name](args=tracker_cfg, frame_rate=30)

        sequence_data_list = []
        # 添加图像处理进度条
        for idx, img in enumerate(tqdm(imgs, desc=f"处理 {seq_name}", leave=False)):
            # Get the image path
            img_path = os.path.join(img_dir, img)

            # Read the image using OpenCV
            img_file = cv2.imread(img_path)
            if img_file is None:
                print(f"警告: 无法读取图像 {img_path}")
                continue

            # Warm up the model
            if idx == 0:
                print(f"  预热模型 (10次推理)...")
                for warmup_idx in range(10):
                    _ = model.predict(
                        source=img_path,
                        verbose=False,
                        save=False,
                        conf=0.1,
                        imgsz=validator.args.imgsz,  # 使用验证器设置的图像尺寸
                        max_det=validator.args.max_det,
                        device=validator.args.device,
                        half=validator.args.half,
                        classes=[0],
                    )[0]
                print(f"  预热完成，开始处理图像...")

            # Infer on the image
            start_time = time.time()
            # 在推理前清理GPU缓存
            torch.cuda.empty_cache()
            result = model.predict(
                source=img_path,
                verbose=False,
                save=False,
                conf=0.1,   # TODO: change to trackers' min confidence
                imgsz=validator.args.imgsz,  # 使用验证器设置的图像尺寸，而不是硬编码的1280
                max_det=min(validator.args.max_det, 200),  # 限制最大检测数量以减少内存使用
                device=validator.args.device,
                half=validator.args.half,
                classes=[0],
            )[0]

            # Process tracker's input
            det = result.boxes.cpu().numpy()

            # Update tracker
            if hasattr(tracker, "with_reid"):
                embeds = result.embeds.data.cpu().numpy()
                tracks = tracker.update(det, img_file, embeds) if tracker.with_reid else tracker.update(det, img_file, None)
                # 立即释放嵌入向量内存
                del embeds
                torch.cuda.empty_cache()  # 强制清理GPU缓存
            else:
                tracks = tracker.update(det, img_file)
            
            # 释放检测结果内存
            del det, result
            torch.cuda.empty_cache()  # 强制清理GPU缓存

            # Update counters
            frame_time = time.time() - start_time
            total_time += frame_time
            total_frames += 1

            # Process results
            if len(tracks) == 0:
                continue
            frame_data = np.hstack([
                (np.ones_like(tracks[:, 0]) * (idx + 1)).reshape(-1, 1),  # Frame number
                tracks[:, 4].reshape(-1, 1),  # Track ID
                tracks[:, :4],  # Bbox XYXY
            ])
            sequence_data_list.append(frame_data)

        # Save results to file
        if len(sequence_data_list) > 0:
            sequence_data = np.vstack(sequence_data_list)

            # Convert Bbox in indices 2:6 from TLBR to TLWH  format
            sequence_data[:, 4] -= sequence_data[:, 2]
            sequence_data[:, 5] -= sequence_data[:, 3]

            # Add confidence, class, visibility and empty columns
            constant_cols = np.ones((sequence_data.shape[0], 4)) * -1
            sequence_data = np.hstack([sequence_data, constant_cols])

            # Save results to file
            txt_path = output_folder + f'/{seq_name}.txt'
            with open(txt_path, 'w') as file:
                np.savetxt(file, sequence_data, fmt='%.6f', delimiter=',')
            
            # 释放序列数据内存
            del sequence_data, constant_cols
        else:
            # 如果没有检测结果，创建空文件
            txt_path = output_folder + f'/{seq_name}.txt'
            with open(txt_path, 'w') as file:
                pass
        
        # 释放序列数据列表内存
        del sequence_data_list
        # 释放图像文件内存
        del img_file

    # Print results
    print(f"Total frames: {total_frames}")
    print(f"Total time (s): {total_time:.3f}")
    print(f"Mean FPS: {total_frames / total_time:.3f}")

    # Evaluate the sequences
    # 确定基准名称（MOT17 或 MOT20）
    benchmark = 'MOT20' if 'MOT20' in dataset_name else 'MOT17'
    
    config = {
        'GT_FOLDER': dataset_root,
        'TRACKERS_FOLDER': '/'.join(output_folder.split('/')[:-1]),
        'TRACKERS_TO_EVAL': [''],
        'METRICS': ['HOTA', 'CLEAR', 'Identity'],
        'USE_PARALLEL': True,
        'NUM_PARALLEL_CORES': 4,
        'SKIP_SPLIT_FOL': True,
        'SEQMAP_FILE': seqmap_file,
        'BENCHMARK': benchmark,  # 设置基准名称
        'SPLIT_TO_EVAL': 'train' if 'MOT20' in dataset_name else 'val_half',  # MOT20 验证集在 train 目录
        'PRINT_CONFIG': False,
        'PRINT_RESULTS': False,
    }

    trackeval_evaluation(config)

    # Read HOTA, MOTA, and IDF1 from summary file
    summary_path = '/'.join(output_folder.split('/')[:-1]) + '/pedestrian_summary.txt'
    summary_df = pd.read_csv(summary_path, sep=' ')
    hota, mota, idf1 = summary_df.loc[0, ['HOTA', 'MOTA', 'IDF1']]
    print(f'HOTA: {hota:.3f}, MOTA: {mota:.3f}, IDF1: {idf1:.3f}')

    # Log metrics
    validator.reid_metrics.set_trackeval_metrics(hota, mota, idf1)


def dict_to_namespace(d):
    return SimpleNamespace(**{k: dict_to_namespace(v) if isinstance(v, dict) else v for k, v in d.items()})
