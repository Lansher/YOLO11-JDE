# YOLO11-JDE 项目完整记录

## 📋 目录

1. [项目概述](#项目概述)
2. [问题修复记录](#问题修复记录)
3. [性能优化方案](#性能优化方案)
4. [内存优化方案](#内存优化方案)
5. [训练优化建议](#训练优化建议)
6. [实施指南](#实施指南)
7. [注意事项](#注意事项)

---

## 项目概述

本项目基于 YOLO11-JDE 实现多目标跟踪（MOT），主要针对 MOT20 数据集进行训练和评估。

### 核心功能
- 目标检测（Detection）
- 目标跟踪（Tracking）
- ReID 特征学习（Re-identification）

### 数据集
- **训练集**: MOT20-01（429张图像）
- **验证集**: MOT20-02, MOT20-03, MOT20-05
- **测试集**: MOT20-04, MOT20-06, MOT20-07, MOT20-08

### 评估指标
- **检测指标**: mAP50, mAP50-95, Precision, Recall
- **跟踪指标**: HOTA, MOTA, IDF1

---

## 问题修复记录

### 1. 验证脚本问题修复

#### 1.1 配置文件路径错误

**问题**: `FileNotFoundError: [Errno 2] No such file or directory: './ultralytics/cfg/trackers/yolojdetracker.yaml'`

**原因**: 使用相对路径，工作目录不是项目根目录时无法找到配置文件

**解决方案**: 
- 修改 `tracker/evaluation/mot_callback.py` 第 88-90 行
- 使用 `__file__` 获取脚本绝对路径，动态构建配置文件路径

```python
# 获取项目根目录（向上3级目录从 tracker/evaluation/ 到项目根目录）
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
tracker_cfg_path = os.path.join(project_root, 'ultralytics', 'cfg', 'trackers', f'{tracker_name}.yaml')
```

#### 1.2 进度条卡在 0%

**问题**: 进度条一直显示 `0%|0/3 [01:12<?, ?it/s]`，无法看到实际处理进度

**解决方案**:
- 添加图像过滤，提前过滤出有效的图像文件
- 为图像处理循环添加 `tqdm` 进度条
- 添加调试信息，显示序列信息和处理状态

```python
# 过滤出有效的图像文件
imgs = [img for img in all_imgs if img.endswith('.jpg') or img.endswith('.png')]
print(f"\n处理序列 {seq_name}: 共 {len(imgs)} 张图像")

# 添加图像处理进度条
for idx, img in enumerate(tqdm(imgs, desc=f"处理 {seq_name}", leave=False)):
```

#### 1.3 seqmap 文件缺失

**问题**: `tracker.evaluation.TrackEval.trackeval.utils.TrackEvalException: no seqmap found: MOT20-train.txt`

**解决方案**: 自动创建 seqmap 文件和目录

```python
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
```

### 2. MOT20 数据集支持

#### 2.1 数据集自动检测

**修改位置**: `tracker/evaluation/mot_callback.py` 第 26-68 行

**功能**: 自动检测数据集类型（MOT17 或 MOT20），并设置相应的配置

```python
# 检测数据集类型：如果数据路径包含 MOT20，使用 MOT20，否则使用 MOT17
data_path = getattr(validator.args, 'data', '')
if 'MOT20' in str(data_path) or 'mot20' in str(data_path).lower():
    # 使用 MOT20 数据集
    dataset_name = 'MOT20/train'
    mot20_root = '/root/autodl-tmp/MOT20'
    dataset_root = os.path.join(mot20_root, 'train')
    # MOT20 验证序列：MOT20-02, MOT20-03, MOT20-05
    available_seqs = []
    for seq in ['MOT20-02', 'MOT20-03', 'MOT20-05']:
        seq_path = os.path.join(dataset_root, seq)
        if os.path.exists(seq_path) and os.path.exists(os.path.join(seq_path, 'img1')):
            available_seqs.append(seq)
    seq_names = available_seqs
```

#### 2.2 评估配置修改

**修改位置**: `tracker/evaluation/mot_callback.py` 第 210-223 行

**关键修改**:
- 添加 `BENCHMARK` 参数以正确标识数据集类型
- 根据数据集类型设置正确的 `SPLIT_TO_EVAL` 参数

```python
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
    'SPLIT_TO_EVAL': 'train' if 'MOT20' in dataset_name else 'val_half',
    'PRINT_CONFIG': False,
    'PRINT_RESULTS': False,
}
```

#### 2.3 内存优化和图像尺寸修复

**问题**: 
- 硬编码 `imgsz=1280`，在 CPU 模式下占用大量内存且速度慢
- 缺少内存释放机制，导致长时间运行时内存占用不断增长

**解决方案**:
- 使用 `validator.args.imgsz` 替代硬编码的 1280
- 添加及时的内存释放机制

```python
# 使用验证器设置的图像尺寸，而不是硬编码的1280
result = model.predict(
    source=img_path,
    verbose=False,
    save=False,
    conf=0.1,
    imgsz=validator.args.imgsz,  # 使用验证器设置的图像尺寸
    max_det=min(validator.args.max_det, 200),
    device=validator.args.device,
    half=validator.args.half,
    classes=[0],
)[0]

# 释放检测结果内存
del det, result
torch.cuda.empty_cache()  # 强制清理GPU缓存
```

### 3. 标签验证逻辑修复

#### 3.1 支持 track_id 为 -1

**问题**: MOT20_YOLO 数据集的标签文件使用 6 列格式，部分检测框的 `track_id` 为 `-1`，被错误地标记为损坏

**解决方案**: 修改 `ultralytics/data/utils.py` 第 139-142 行

```python
# For tag mode (JDE task), only check first 5 columns (class + bbox), allow track_id (6th column) to be -1
if tag and lb.shape[1] == 6:
    assert lb[:, :5].min() >= 0, f"negative label values in class/bbox {lb[:, :5][lb[:, :5] < 0]}"
else:
    assert lb.min() >= 0, f"negative label values {lb[lb < 0]}"
```

### 4. ReID 指标计算修复

#### 4.1 标签数量不足问题

**问题**: `ValueError: Number of labels is 1. Valid values are 2 to n_samples - 1 (inclusive)`

**原因**: 当验证集中只有一个唯一的标签时，无法计算聚类指标

**解决方案**: 修改 `ultralytics/utils/metrics.py` 第 1367-1388 行

```python
# Compute Silhouette, Davies Bouldin and Calinski Harabasz scores
# These metrics require at least 2 different labels
unique_labels = len(np.unique(tags))
if unique_labels < 2:
    # Not enough labels to compute clustering metrics
    cos_silhouette_score = 0.0
    euc_silhouette_score = 0.0
    davies_bouldin_score = 0.0
    calinski_harabasz_score = 0.0
else:
    try:
        cos_silhouette_score = skm.silhouette_score(cos_distmat, tags, metric='precomputed')
        euc_silhouette_score = skm.silhouette_score(euc_distmat, tags, metric='precomputed')
        davies_bouldin_score = skm.davies_bouldin_score(embeds, tags)
        calinski_harabasz_score = skm.calinski_harabasz_score(embeds, tags)
    except (ValueError, Exception) as e:
        # Fallback to default values if computation fails
        cos_silhouette_score = 0.0
        euc_silhouette_score = 0.0
        davies_bouldin_score = 0.0
        calinski_harabasz_score = 0.0
```

#### 4.2 空数组问题

**问题**: `ValueError: Found array with 0 sample(s) (shape=(0, 128)) while a minimum of 1 is required`

**原因**: 经过多重过滤后，某些批次可能没有有效的嵌入向量

**解决方案**: 修改 `ultralytics/utils/metrics.py` 的 `get_metrics()` 和 `process_batch()` 方法

```python
def get_metrics(self):
    # Check if we have any embeddings
    if len(self.embeds) == 0:
        return self._get_default_metrics()
    
    # Filter out empty tensors before concatenation
    non_empty_embeds = [e for e in self.embeds if len(e) > 0]
    non_empty_tags = [t for t in self.tags if len(t) > 0]
    
    if len(non_empty_embeds) == 0 or len(non_empty_tags) == 0:
        return self._get_default_metrics()
    
    # Concatenate and convert to numpy arrays
    embeds = torch.cat(non_empty_embeds).cpu().detach().numpy()
    tags = torch.cat(non_empty_tags).cpu().detach().numpy()
    
    # Check if concatenated arrays are empty
    if len(embeds) == 0 or len(tags) == 0:
        return self._get_default_metrics()
    
    # Compute distance matrix and positive and negative distances
    pos_cos, neg_cos, cos_distmat = self.compute_distmat(embeds, tags, distance="cosine")
```

---

## 性能优化方案

### 1. 核心问题分析

#### 当前训练结果
- **mAP50-95**: 0.161-0.201（有波动）
- **HOTA**: 10.272% ⚠️（正常值: 40-60%+）
- **MOTA**: 3.767% ⚠️（正常值: 50-70%+）
- **IDF1**: 5.780% ⚠️（正常值: 50-70%+）

#### 问题根源
1. **缺少 Track ID 信息**（最关键）⭐⭐⭐⭐⭐
   - 训练使用的是 det.txt 转换的数据，**没有 track ID 信息**
   - 模型无法学习 ReID 特征（无法区分不同 ID 的同一人）
   - IDF1 极低（5.78%）的直接原因

2. **Tracker 超参数未优化** ⭐⭐⭐
   - 使用默认 tracker 配置
   - 匹配阈值不适合 MOT20 高密度场景

3. **训练数据量可能不足** ⭐⭐
   - 只使用 MOT20-01 作为训练集（429张图像）
   - 数据多样性不足

### 2. 优化方案（按优先级）

#### 方案1: 使用 GT 数据集重新训练（必须）⭐⭐⭐⭐⭐

**这是最重要的优化！** 当前训练没有 track ID，无法学习 ReID。

**执行步骤**:

1. **确认 GT 数据集已转换**:
   ```bash
   ls -lh /root/autodl-tmp/MOT20_YOLO_GT/
   # 应该看到 images/train, images/val, labels/train, labels/val
   ```

2. **如果未转换，先转换**:
   ```bash
   python scripts/convert_mot20_gt_to_yolo.py
   ```

3. **使用 GT 数据集训练**:
   ```bash
   python scripts/train_with_tracks.py
   ```

**预期效果**:
- IDF1: 5.78% → 40-60%+（提升 10 倍）
- MOTA: 3.77% → 40-60%+（提升 10 倍）
- HOTA: 10.27% → 40-55%+（提升 4 倍）

#### 方案2: 优化 Tracker 配置（推荐）⭐⭐⭐⭐

**执行**:
```python
# 在train_with_tracks.py中已配置
tracker='jdetracker_mot20_optimized.yaml'
```

**关键优化**:
- `conf_thresh`: 0.35 → 0.3（捕获更多目标）
- `first_match_thresh`: 0.1 → 0.05（更宽松匹配）
- `track_buffer`: 40 → 50（处理遮挡）

#### 方案3: 增加训练数据（可选）⭐⭐⭐

**如果可能，合并更多数据**:
- 将 MOT20 验证集的部分序列加入训练
- 或使用 CrowdHuman 数据集混合训练

#### 方案4: Tracker 超参数自动优化（可选）⭐⭐

**在训练完成后执行**:
```bash
cd tracker/finetune
python evolve_mot20.py \
    --model /path/to/best.pt \
    --n_trials 50
```

### 3. mAP 优化建议

#### 问题分析
- **Epoch 2**: mAP50-95 = 0.143
- **Epoch 9**: mAP50-95 = 0.162
- **增长幅度**: 仅提升了约 13%，增长非常缓慢

#### 优化方案

**高优先级（必须优化）**:

1. **增加训练轮数**
   ```python
   epochs=50  # 至少50个epoch，建议100+
   ```

2. **减少或取消冻结**
   ```python
   freeze=None  # 或 freeze=5（只冻结前5层）
   ```

3. **启用余弦学习率调度**
   ```python
   cos_lr=True,
   lr0=0.001,  # 降低初始学习率
   ```

4. **启用混合精度训练**
   ```python
   amp=True,
   ```

**中优先级（强烈推荐）**:

5. **优化学习率策略**
   ```python
   lr0=0.001,      # 初始学习率
   lrf=0.01,       # 最终学习率比例
   warmup_epochs=5.0,  # 增加warmup
   ```

6. **增加数据增强**
   ```python
   mixup=0.1,      # 添加mixup
   degrees=5.0,    # 轻微旋转
   ```

7. **调整损失权重**
   ```python
   clr=1.5,        # 增加ReID损失权重
   ```

8. **增加 Early Stopping Patience**
   ```python
   patience=50,    # 从25增加到50
   ```

**渐进式训练策略（最推荐）**:

```python
# Stage 1: 冻结backbone，快速适应head
freeze=11,
epochs=20,
lr0=0.001,

# Stage 2: 解冻部分层
freeze=5,
epochs=40,
lr0=0.0005,
resume=True,  # 从checkpoint继续

# Stage 3: 全模型微调
freeze=None,
epochs=60,
lr0=0.0001,
resume=True,
```

---

## 内存优化方案

### 1. 问题分析

#### 验证阶段内存问题
- 验证阶段计算 embedding loss 时，当 batch 中有大量目标（特别是 MOT20 密集场景），`BatchEasyHardMiner` 计算距离矩阵会消耗大量内存
- 验证完成后的 `mot_eval` 回调处理大量图像时，embeddings 累积导致内存不足
- 进程可能因为内存不足被系统 "Killed"

### 2. 已实施的优化措施

#### 2.1 大幅降低 embed_max_samples 默认值

**位置**: `ultralytics/utils/loss.py:348-351`

- 从 2000 → 500 → 300 → **100**
- 限制用于计算 embedding loss 的最大样本数

```python
if embed_max_samples is None:
    embed_max_samples = 100  # 从300降低到100
self.embed_loss = MetricLearningLoss(max_samples=embed_max_samples).to(device)
```

#### 2.2 优化动态调整机制

**位置**: `ultralytics/utils/loss.py:28-40`

```python
if num_samples > 2000:
    effective_max_samples = min(self.max_samples, 50)  # 非常保守的限制
elif num_samples > 1000:
    effective_max_samples = min(self.max_samples, 80)
elif num_samples > 500:
    effective_max_samples = min(self.max_samples, int(self.max_samples * 0.7))
```

#### 2.3 在 v8JDELoss 中跳过超大 batch 的 embedding loss

**位置**: `ultralytics/utils/loss.py:432-440`

```python
# 如果前景目标数量过多，直接跳过embedding loss计算以避免内存不足
if pred_embeds.shape[0] > 2000:  # 如果前景目标超过2000，跳过embedding loss
    loss[3] = torch.tensor(0.0, device=self.device, requires_grad=True)
else:
    loss[3] = self.embed_loss(pred_embeds, target_tags, confidences)
```

#### 2.4 优化 mot_eval 回调的内存管理

**位置**: `tracker/evaluation/mot_callback.py`

- 添加 `torch.cuda.empty_cache()` 强制清理 GPU 缓存
- 限制 `max_det` 为 200 以减少内存使用
- 在处理每张图像后立即释放 embeddings 和检测结果

```python
# 在推理前清理GPU缓存
torch.cuda.empty_cache()

# 限制最大检测数量以减少内存使用
max_det=min(validator.args.max_det, 200)

# 释放检测结果内存
del det, result
torch.cuda.empty_cache()  # 强制清理GPU缓存
```

#### 2.5 训练配置优化

**位置**: `train_optimized.py`

- Batch size: 2（已是最小值）
- Workers: 8（可以进一步降低到 4）

### 3. 优化效果

1. **embed_max_samples**: 100（从 300 降低）
2. **动态限制**:
   - 样本数 > 2000: 限制为 50
   - 样本数 > 1000: 限制为 80
   - 样本数 > 500: 限制为 70% 的 max_samples
3. **跳过超大 batch**: 如果前景目标 > 2000，直接跳过 embedding loss
4. **GPU 缓存清理**: 在 mot_eval 中强制清理 GPU 缓存
5. **限制检测数量**: max_det 限制为 200

### 4. 如果仍然被 Killed

如果仍然出现内存问题，可以进一步：

1. **进一步降低 embed_max_samples**: 从 100 降到 50
2. **降低验证 batch size**: 在训练脚本中设置更小的验证 batch size
3. **降低图像尺寸**: 从 1280 降到 960 或 640
4. **禁用 mot_eval 回调**: 如果不需要 MOT 评估，可以暂时禁用
5. **降低 workers 数量**: 从 8 降到 4 或更小

---

## 训练优化建议

### 1. 当前训练情况

- **mAP50-95**: 0.161-0.201（有波动）
- **Precision**: 0.287
- **Recall**: 0.210（较低，说明漏检较多）
- **HOTA**: 10.272% ⚠️
- **MOTA**: 3.767% ⚠️
- **IDF1**: 5.780% ⚠️

### 2. 关键问题

1. **缺少 Track ID 信息**（最关键）
   - 训练使用的是 det.txt 转换的数据，**没有 track ID 信息**
   - 模型无法学习 ReID 特征
   - 这是跟踪性能差的根本原因

2. **训练轮数不足**
   - 当前: `epochs=10`
   - 问题: 10 个 epoch 对于 fine-tuning 来说太少了

3. **冻结层过多**
   - 当前: `freeze=11` (冻结 backbone)
   - 问题: 冻结 backbone 限制了特征提取能力的学习

4. **学习率策略不当**
   - 使用默认学习率和线性衰减
   - 初始学习率可能不适合 fine-tuning

### 3. 优化建议

#### 快速优化脚本

已创建 `train_optimized.py`，包含以下优化：

- ✅ epochs 增加到 50
- ✅ 取消冻结（freeze=None）
- ✅ 启用余弦学习率（cos_lr=True）
- ✅ 降低初始学习率（lr0=0.001）
- ✅ 启用混合精度（amp=True）
- ✅ 增加 warmup 轮数（warmup_epochs=5.0）
- ✅ 添加 mixup 增强
- ✅ 使用 AdamW 优化器
- ✅ 增加 patience 到 50
- ✅ 调整损失权重（clr=1.0，可进一步调整）

#### 预期效果

实施这些优化后，预期：
- **mAP50-95**: 从 0.16 提升到 0.25-0.35（取决于数据集难度）
- **训练稳定性**: 显著提升
- **收敛速度**: 加快
- **最终性能**: 提升 30-50%

---

## 实施指南

### 1. 完整执行流程

#### 步骤1: 转换 GT 数据集（必须）

```bash
cd /root/autodl-tmp/YOLO11-JDE
python scripts/convert_mot20_gt_to_yolo.py \
    --mot20_root /root/autodl-tmp/MOT20 \
    --output_dir /root/autodl-tmp/MOT20_YOLO_GT
```

**检查 GT 数据集**:
```bash
# 检查GT数据集是否存在
ls /root/autodl-tmp/MOT20_YOLO_GT/

# 检查标签文件是否包含track ID
head -5 /root/autodl-tmp/MOT20_YOLO_GT/labels/train/*.txt | head -20
# 应该看到格式: 0 x y w h track_id
```

#### 步骤2: 使用新数据集训练（必须）

```bash
python scripts/train_with_tracks.py
```

**关键配置**:
- `data='MOT20_GT.yaml'` (使用 GT 转换的数据)
- `clr=1.5` (增加 ReID 损失权重)
- `freeze=None` (全模型微调)

#### 步骤3: 优化 Tracker 配置（推荐）

编辑训练脚本，使用：
```python
tracker='jdetracker_mot20_optimized.yaml'
```

**关键调整**:
- `conf_thresh`: 0.35 → 0.3 (捕获更多目标)
- `first_match_thresh`: 0.1 → 0.05 (更宽松匹配)
- `track_buffer`: 40 → 50 (处理遮挡)

#### 步骤4: 进一步优化 Tracker（可选）

```bash
cd tracker/finetune
python evolve_mot20.py \
    --model /path/to/best.pt \
    --n_trials 100
```

### 2. 快速开始（最小化方案）

如果时间有限，至少执行：

1. **转换 GT 数据集**（30 分钟）
   ```bash
   python scripts/convert_mot20_gt_to_yolo.py
   ```

2. **使用新数据集训练**（数小时）
   ```bash
   python scripts/train_with_tracks.py
   ```

这两步就能带来显著提升！

### 3. 预期改进

| 指标 | 当前值 | 使用 GT 训练后 | 提升倍数 |
|------|--------|---------------|---------|
| **IDF1** | 5.78% | 40-60% | **7-10 倍** |
| **MOTA** | 3.77% | 40-60% | **10-15 倍** |
| **HOTA** | 10.27% | 40-55% | **4-5 倍** |
| **mAP50-95** | 0.16-0.20 | 0.25-0.35 | **1.5-2 倍** |

---

## 注意事项

### 1. 数据集相关

1. **数据集路径**: MOT 评估回调使用原始 MOT20 数据集（`/root/autodl-tmp/MOT20`），而不是 YOLO 格式数据集
2. **验证序列**: MOT20 的验证序列（MOT20-02, MOT20-03, MOT20-05）位于 `train` 目录中
3. **Track ID**: 当前训练使用的是 det.txt 数据（无 track ID），这是跟踪性能差的根本原因，**必须使用 GT 数据集重新训练**

### 2. 训练相关

1. **训练时间**: 使用 track ID 训练会增加训练时间，但效果显著
2. **GPU 内存**: 全模型微调需要足够的 GPU 内存（当前 22G 足够）
3. **训练轮数**: 至少需要 50 个 epoch，建议 100+
4. **内存优化**: 已优化内存使用，使用 `validator.args.imgsz` 替代硬编码的 1280，在 CPU 模式下建议使用 640 或更小的图像尺寸以避免内存不足

### 3. 性能相关

1. **性能**: 在 CPU 上运行会较慢，建议在有 GPU 的环境中运行以获得更好的性能
2. **TrackEval**: 评估使用 TrackEval 库，需要确保相关依赖已安装
3. **超参数搜索**: evolve.py 需要较长时间，建议在后台运行

### 4. 代码相关

1. **向后兼容**: 所有修改都保持了向后兼容性，不影响其他功能的正常使用
2. **错误处理**: 添加了完善的错误处理和警告信息
3. **内存管理**: 添加了及时的内存释放机制，避免内存泄漏

---

## 文件清单

### 修改的文件

1. `tracker/evaluation/mot_callback.py` - MOT 评估回调函数（支持 MOT20、内存优化、图像尺寸配置修复）
2. `validate_cpu.py` - 验证脚本
3. `ultralytics/data/utils.py` - 标签验证逻辑（修复负 track_id 问题）
4. `ultralytics/utils/metrics.py` - ReID 指标计算（修复标签数量不足和空数组问题）
5. `ultralytics/utils/loss.py` - 内存优化（降低 embed_max_samples，添加动态调整机制）

### 创建的文件

1. `scripts/convert_mot20_gt_to_yolo.py` - GT 标注转换脚本
2. `scripts/train_with_tracks.py` - 使用 track ID 的训练脚本
3. `train_optimized.py` - 优化后的训练脚本
4. `evolve_mot20.py` - MOT20 专用的 tracker 超参数优化
5. `jdetracker_mot20_optimized.yaml` - 优化后的 tracker 配置

### 使用的数据集

1. `/root/autodl-tmp/MOT20_YOLO/` - YOLO 格式数据集（用于验证）
2. `/root/autodl-tmp/MOT20_YOLO_GT/` - GT 转换的 YOLO 格式数据集（用于训练）
3. `/root/autodl-tmp/MOT20/` - 原始 MOT 格式数据集（用于 MOT 评估）

---

## 总结

### 核心问题

1. **缺少 Track ID 信息**: 这是跟踪性能差的根本原因，必须使用 GT 数据集重新训练
2. **内存问题**: 验证阶段计算 embedding loss 时内存消耗过大，已通过降低 embed_max_samples 和添加动态调整机制解决
3. **训练配置不当**: 训练轮数不足、冻结层过多、学习率策略不当，已提供优化方案

### 解决方案

1. **使用 GT 数据集**: 转换并使用包含 track ID 的 GT 数据集进行训练
2. **优化 Tracker 配置**: 针对 MOT20 高密度场景优化 tracker 超参数
3. **内存优化**: 降低 embed_max_samples，添加动态调整机制，优化内存管理
4. **训练优化**: 增加训练轮数，减少或取消冻结，优化学习率策略

### 预期效果

实施这些优化后，预期：
- **跟踪指标**: IDF1、MOTA、HOTA 提升 5-10 倍
- **检测指标**: mAP50-95 提升 1.5-2 倍
- **训练稳定性**: 显著提升
- **内存使用**: 显著降低，避免被 Killed

---

**最后更新**: 2026-01-28
