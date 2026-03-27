"""
针对MOT20数据集的Tracker超参数优化脚本
基于evolve.py，但针对MOT20的高密度场景进行了优化
"""

import os
import random
import string
import time
from types import SimpleNamespace

import cv2
import numpy as np
import optuna
import joblib
import yaml

from ultralytics import YOLO
from tracker.evaluation.evaluate import trackeval_evaluation
from ultralytics.trackers import YOLOJDETracker


def generate_unique_tag():
    """生成唯一的实验标签"""
    timestamp = int(time.time())
    random_suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
    tag = f"exp_{timestamp}_{random_suffix}"
    return tag


def dict_to_namespace(d):
    """将字典转换为SimpleNamespace"""
    return SimpleNamespace(**{k: dict_to_namespace(v) if isinstance(v, dict) else v for k, v in d.items()})


def optuna_fitness_fn(trial, model_path, dataset_root, seq_names, device=[0]):
    """
    Optuna优化目标函数 - 针对MOT20优化
    
    Args:
        trial: Optuna trial对象
        model_path: 模型路径
        dataset_root: 数据集根目录
        seq_names: 序列名称列表
        device: GPU设备列表
    """
    output_folder = f'./outputs/' + generate_unique_tag() + '/MOT20/train/data'
    os.makedirs(output_folder, exist_ok=True)

    # 加载模型
    model = YOLO(model_path, task='jde')

    # 加载基础配置
    config_path = os.path.join('./../../ultralytics/cfg/trackers/jdetracker.yaml')
    tracker_config = dict_to_namespace(yaml.safe_load(open(config_path)))

    # 定义超参数搜索空间（针对MOT20高密度场景优化）
    tracker_config.track_buffer = trial.suggest_int("track_buffer", 30, 80, step=10)
    tracker_config.conf_thresh = trial.suggest_float("conf_thresh", 0.2, 0.5, step=0.05)
    
    # 匹配阈值（MOT20需要更宽松的首次匹配）
    tracker_config.first_match_thresh = trial.suggest_float("first_match_thresh", 0.0, 0.3, step=0.05)
    tracker_config.second_match_thresh = trial.suggest_float("second_match_thresh", 0.6, 0.95, step=0.05)
    tracker_config.new_match_thresh = trial.suggest_float("new_match_thresh", 0.5, 0.85, step=0.05)

    # ReID相关参数（如果tracker支持）
    if hasattr(tracker_config, 'appearance_thresh'):
        tracker_config.appearance_thresh = trial.suggest_float("appearance_thresh", 0.1, 0.5, step=0.05)
    if hasattr(tracker_config, 'gate_thresh'):
        tracker_config.gate_thresh = trial.suggest_float("gate_thresh", 0.1, 0.5, step=0.05)

    # 处理每个序列
    for seq_name in seq_names:
        seq_path = os.path.join(dataset_root, seq_name)
        img_folder = os.path.join(seq_path, 'img1')
        
        if not os.path.exists(img_folder):
            print(f"警告: 序列 {seq_name} 的图像文件夹不存在，跳过")
            continue

        # 排序图像
        imgs = sorted([f for f in os.listdir(img_folder) if f.endswith(('.jpg', '.png'))])

        # 初始化tracker（每个序列重新初始化）
        # 从seqinfo.ini读取帧率，默认25fps
        seqinfo_file = os.path.join(seq_path, 'seqinfo.ini')
        frame_rate = 25.0
        if os.path.exists(seqinfo_file):
            with open(seqinfo_file, 'r') as f:
                for line in f:
                    if line.startswith('frameRate'):
                        frame_rate = float(line.split('=')[1].strip())
                        break
        
        tracker = YOLOJDETracker(args=tracker_config, frame_rate=frame_rate)

        sequence_data_list = []
        for idx, img_name in enumerate(imgs):
            img_path = os.path.join(img_folder, img_name)

            # 读取图像
            img_file = cv2.imread(img_path)
            if img_file is None:
                continue

            # 模型推理
            result = model.predict(
                source=img_path,
                verbose=False,
                save=False,
                conf=0.1,  # 低置信度阈值以捕获更多目标
                imgsz=1280,
                max_det=300,
                device=device,
                half=False,
                classes=[0],  # 只检测person
            )[0]

            # 处理检测结果
            det = result.boxes.cpu().numpy()

            # 更新tracker
            if hasattr(tracker, "with_reid") and tracker.with_reid:
                embeds = result.embeds.data.cpu().numpy() if hasattr(result, 'embeds') else None
                tracks = tracker.update(det, img_file, embeds)
            else:
                tracks = tracker.update(det, img_file)

            # 保存跟踪结果
            if len(tracks) > 0:
                frame_data = np.hstack([
                    (np.ones_like(tracks[:, 0]) * (idx + 1)).reshape(-1, 1),  # Frame number
                    tracks[:, 4].reshape(-1, 1),  # Track ID
                    tracks[:, :4],  # Bbox XYXY
                ])
                sequence_data_list.append(frame_data)

        # 保存序列结果
        if len(sequence_data_list) > 0:
            sequence_data = np.vstack(sequence_data_list)
        else:
            sequence_data = np.zeros((0, 6))

        # 转换bbox格式：从TLBR到TLWH
        sequence_data[:, 4] -= sequence_data[:, 2]
        sequence_data[:, 5] -= sequence_data[:, 3]

        # 添加必要的列（MOT格式要求）
        constant_cols = np.ones((sequence_data.shape[0], 4)) * -1
        sequence_data = np.hstack([sequence_data, constant_cols])

        # 保存到文件
        txt_path = os.path.join(output_folder, f'{seq_name}.txt')
        with open(txt_path, 'w') as file:
            np.savetxt(file, sequence_data, fmt='%.6f', delimiter=',')

    # 评估结果
    seqmap_file = os.path.join(dataset_root, 'seqmaps', 'MOT20-train.txt')
    if not os.path.exists(seqmap_file):
        # 创建seqmap文件
        seqmap_dir = os.path.join(dataset_root, 'seqmaps')
        os.makedirs(seqmap_dir, exist_ok=True)
        with open(seqmap_file, 'w') as f:
            f.write('name\n')
            for seq in seq_names:
                f.write(f'{seq}\n')

    config = {
        'GT_FOLDER': dataset_root,
        'TRACKERS_FOLDER': '/'.join(output_folder.split('/')[:-1]),
        'TRACKERS_TO_EVAL': [''],
        'METRICS': ['HOTA', 'CLEAR', 'Identity'],
        'USE_PARALLEL': True,
        'NUM_PARALLEL_CORES': 4,
        'SKIP_SPLIT_FOL': True,
        'SEQMAP_FILE': seqmap_file,
        'PRINT_CONFIG': False,
        'PRINT_RESULTS': False,
    }

    try:
        trackeval_evaluation(config)
        
        # 读取评估结果
        summary_path = '/'.join(output_folder.split('/')[:-1]) + '/pedestrian_summary.txt'
        if os.path.exists(summary_path):
            import pandas as pd
            summary_df = pd.read_csv(summary_path, sep=' ')
            hota = summary_df.loc[0, 'HOTA']
            return hota
        else:
            print(f"警告: 评估结果文件不存在: {summary_path}")
            return 0.0
    except Exception as e:
        print(f"评估出错: {e}")
        return 0.0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='MOT20 Tracker超参数优化')
    parser.add_argument('--model', type=str, 
                        default='./../../reid_xps/CH-jde-16b-50e_OPTIMIZED_TBHS_m075_1280px_20260127-132420/weights/best.pt',
                        help='模型路径')
    parser.add_argument('--dataset_root', type=str,
                        default='/root/autodl-tmp/MOT20/train',
                        help='MOT20数据集根目录（train文件夹）')
    parser.add_argument('--seqs', nargs='+', 
                        default=['MOT20-02', 'MOT20-03', 'MOT20-05'],
                        help='要评估的序列列表')
    parser.add_argument('--device', type=int, nargs='+', default=[0],
                        help='GPU设备列表')
    parser.add_argument('--n_trials', type=int, default=100,
                        help='优化试验次数')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复之前的study（.pkl文件路径）')
    
    args = parser.parse_args()
    
    # 检查序列是否存在
    available_seqs = []
    for seq in args.seqs:
        seq_path = os.path.join(args.dataset_root, seq)
        if os.path.exists(seq_path):
            available_seqs.append(seq)
        else:
            print(f"警告: 序列 {seq} 不存在，跳过")
    
    if len(available_seqs) == 0:
        print("错误: 没有可用的序列！")
        exit(1)
    
    print(f"将优化以下序列: {available_seqs}")
    
    # 创建或加载study
    if args.resume and os.path.exists(args.resume):
        print(f"恢复study: {args.resume}")
        study = joblib.load(args.resume)
    else:
        study = optuna.create_study(direction="maximize")
        
        # 加载默认配置作为起始点
        config_path = os.path.join('./../../ultralytics/cfg/trackers/jdetracker.yaml')
        config = dict_to_namespace(yaml.safe_load(open(config_path)))
        
        # 添加默认参数作为起始试验
        default_params = {
            'track_buffer': getattr(config, 'track_buffer', 40),
            'conf_thresh': getattr(config, 'conf_thresh', 0.35),
            'first_match_thresh': getattr(config, 'first_match_thresh', 0.1),
            'second_match_thresh': getattr(config, 'second_match_thresh', 0.85),
            'new_match_thresh': getattr(config, 'new_match_thresh', 0.7),
        }
        if hasattr(config, 'appearance_thresh'):
            default_params['appearance_thresh'] = getattr(config, 'appearance_thresh', 0.25)
        if hasattr(config, 'gate_thresh'):
            default_params['gate_thresh'] = getattr(config, 'gate_thresh', 0.25)
        
        study.enqueue_trial(default_params)
        print("已添加默认配置作为起始试验")
    
    # 创建fitness函数
    fitness_fn = lambda trial: optuna_fitness_fn(
        trial, args.model, args.dataset_root, available_seqs, args.device
    )
    
    # 运行优化
    print(f"开始优化，共 {args.n_trials} 次试验...")
    study.optimize(
        func=fitness_fn,
        n_trials=args.n_trials,
        show_progress_bar=True,
    )
    
    # 保存study
    study_file = f"mot20_study_{int(time.time())}.pkl"
    joblib.dump(study, study_file)
    print(f"\nStudy已保存到: {study_file}")
    
    # 打印结果
    print("\n" + "="*50)
    print("优化结果:")
    print("="*50)
    print(f"最佳试验编号: {study.best_trial.number}")
    print(f"最佳HOTA值: {study.best_value:.4f}")
    print("\n最佳超参数:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # 生成优化后的配置文件
    config_path = os.path.join('./../../ultralytics/cfg/trackers/jdetracker.yaml')
    config = dict_to_namespace(yaml.safe_load(open(config_path)))
    
    # 更新配置
    for key, value in study.best_params.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # 保存优化后的配置
    optimized_config_path = f"jdetracker_mot20_optimized.yaml"
    optimized_config = {}
    for key in dir(config):
        if not key.startswith('_'):
            optimized_config[key] = getattr(config, key)
    
    with open(optimized_config_path, 'w') as f:
        yaml.dump(optimized_config, f, default_flow_style=False)
    
    print(f"\n优化后的配置文件已保存到: {optimized_config_path}")
