# YOLO11-JDE：基于自监督 Re-ID 的高效多目标跟踪

YOLO11-JDE 将实时检测与 Re-ID 嵌入学习统一到一个端到端框架中（JDE），在保持较高跟踪精度的同时提供有竞争力的实时性能。项目核心特征是：自监督 Re-ID 训练、多线索数据关联（外观+运动+位置）与轻量化部署友好设计。

**论文信息：** WACV 2025 Workshop（Real-World Surveillance: Applications and Challenges），[arXiv:2501.13710](https://arxiv.org/abs/2501.13710v1)。

---

## 目录

- [关键特性](#关键特性)
- [仓库结构](#仓库结构)
- [数据集与标注格式](#数据集与标注格式)
- [模型权重](#模型权重)
- [基准结果](#基准结果)
- [自定义场景优化建议](#自定义场景优化建议)
- [架构与训练推理说明（整合版）](#架构与训练推理说明整合版)
- [项目问题修复与优化记录（整合版）](#项目问题修复与优化记录整合版)
- [量化与推理加速（整合版）](#量化与推理加速整合版)
- [致谢](#致谢)
- [引用](#引用)

---

## 关键特性

- **实时性能**：在 MOT17/MOT20 上达到有竞争力的 FPS 与跟踪精度。
- **自监督 Re-ID**：通过数据增强与 triplet loss（hard / semi-hard mining）降低身份标注依赖。
- **鲁棒关联**：融合外观、运动和位置线索，提升遮挡场景表现。
- **轻量设计**：相较同类 JDE 方法参数量显著更低，易于工程部署。

---

## 仓库结构

- `scripts/`：训练、验证、跟踪、转换等入口脚本（根目录多为轻包装）。
- `ultralytics/`：模型、损失、预测器、跟踪器等核心实现。
- `tracker/`：评估、微调、TrackEval 集成。
- `docs/`：架构分析、项目记录、量化加速等文档。
- `logs/`：离线训练输出目录（git 忽略，保留 `.gitkeep`）。

---

## 数据集与标注格式

训练常用数据集：

1. **CrowdHuman**：拥挤场景目标检测数据。
2. **MOT17 / MOT20**：多目标跟踪序列数据。

数据需转换为 YOLO 格式。JDE 场景下可使用 6 列标签（末列为 `track_id/tag_id`）进行 Re-ID 学习：

```text
class cx cy w h tag_id
```

`track_id/tag_id` 在无监督或弱监督设置下可按训练策略选择是否使用；用于验证/跟踪评估时建议保留。

---

## 模型权重

预训练权重（YOLO11s-JDE）下载：
[Google Drive](https://drive.google.com/drive/folders/16btXRPikwXOsaItn06p4A8cBhTiIJdoZ?usp=share_link)

---

## 基准结果

私有检测协议下结果：

| 指标 | MOT17 | MOT20 |
|---|---:|---:|
| HOTA | 56.6 | 53.1 |
| MOTA | 65.8 | 70.9 |
| IDF1 | 70.3 | 66.4 |
| FPS | 35.9 | 18.9 |

---

## 自定义场景优化建议

1. 调整 `ultralytics/cfg/trackers/yolojdetracker.yaml` 参数。  
2. 通过同域数据减少 domain gap（至少框标注）。  
3. 若具备轨迹信息，使用带 `track_id` 的训练数据。  
4. 使用 `tracker/finetune/evolve.py` 做 tracker 参数进化。

---

## 架构与训练推理说明（整合版）

本节整合自 `docs/YOLO11-JDE架构分析.md`，已去除与前文重复内容。

### 系统分层

```text
JDEModel (Backbone+Neck+JDE Head)
  ├─ 训练：v8JDELoss（box/cls/dfl/emb）
  ├─ 推理：JDEPredictor（NMS + embeds）
  └─ 跟踪：JDETracker（外观+运动+IoU 多阶段关联）
```

### JDE Head 要点

- 在检测头基础上增加嵌入分支（默认 128 维 embedding）。
- 多尺度输出统一拼接，推理时输出检测结果与嵌入特征。

```python
class JDE(Detect):
    # ... detection branches
    # cv4: embedding branch
```

### 损失函数

```text
L_total = λ_box * L_box + λ_cls * L_cls + λ_dfl * L_dfl + λ_clr * L_embed
```

- `L_embed` 使用 triplet loss + hard/semi-hard mining。
- 支持 tag-aware 的目标分配与前景筛选。

### 跟踪关联策略

1. 嵌入距离 + 运动融合（Kalman gating）。  
2. 未匹配目标走 IoU 二次关联。  
3. 余下高置信检测初始化新轨迹。

---

## 项目问题修复与优化记录（整合版）

本节整合自 `docs/YOLO11-JDE项目完整记录.md`，保留关键结论与可执行建议。

### 主要修复点

- `mot_callback.py` 路径与 seqmap 生成问题。
- MOT20 自动识别与评估配置适配。
- 标签合法性校验（允许 `track_id=-1`）。
- ReID 指标在小样本/空数组下的健壮性处理。

### 性能瓶颈结论

- 跟踪指标偏低的首要原因通常是：**训练数据缺少 track id**。  
- 建议优先使用 GT 转换数据重训，并结合 MOT20 场景调优 tracker。

### 内存优化结论

- 限制 embedding loss 样本数（`embed_max_samples`）。  
- 大 batch 条件下跳过或降级 embedding loss。  
- 评估阶段及时释放缓存并限制 `max_det`。  
- CPU 模式建议降低 `imgsz`。

### 预期改进（原记录）

| 指标 | 当前值 | 使用 GT 训练后 | 提升倍数 |
|---|---:|---:|---:|
| IDF1 | 5.78% | 40-60% | 7-10x |
| MOTA | 3.77% | 40-60% | 10-15x |
| HOTA | 10.27% | 40-55% | 4-5x |
| mAP50-95 | 0.16-0.20 | 0.25-0.35 | 1.5-2x |

---

## 量化与推理加速（整合版）

本节整合自 `docs/YOLO11s_JDE_量化加速说明.md`。

### 推荐路径

```text
PyTorch FP32 -> ONNX FP32 -> TensorRT FP32 -> TensorRT FP16
```

若仍需极限加速，可尝试 TensorRT INT8（需校准集，并重点回归跟踪指标）。

### 核心注意事项

- 必须显式使用 `task="jde"`。
- `imgsz` 尽量与训练保持一致。
- INT8 对 Re-ID 嵌入影响可能明显大于 FP16。

### 导出示例

```python
from ultralytics import YOLO

model = YOLO("YOLO11s_JDE-CHMOT17-64b-100e_TBHS_m075_1280px.pt", task="jde")
model.export(format="engine", device=0, half=True, imgsz=1280, batch=1)
```

```bash
yolo mode=export model=YOLO11s_JDE-CHMOT17-64b-100e_TBHS_m075_1280px.pt \
  format=engine device=0 half=True imgsz=1280 batch=1 task=jde
```

### 验收清单

| 项目 | 检查项 |
|---|---|
| 正确性 | FP32/FP16/INT8 在同视频上的检测与 ID 是否可接受 |
| 性能 | 固定分辨率与 batch，预热后统计延迟/FPS |
| 嵌入质量 | INT8 是否造成 ID 切换显著增多 |
| 环境一致性 | CUDA / TensorRT / 驱动版本固定 |

---

## 致谢

本工作部分受以下项目支持：

- Spanish project PID2022-136436NB-I00
- ICREA Academia programme
- Milestone Research Program at the University of Barcelona

项目基于 [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) 实现。

---

## 引用

如果该项目对你的研究或工程有帮助，请引用：

```bibtex
@inproceedings{Erregue_2025,
   title={YOLO11-JDE: Fast and Accurate Multi-Object Tracking with Self-Supervised Re-ID},
   url={http://dx.doi.org/10.1109/WACVW65960.2025.00092},
   DOI={10.1109/wacvw65960.2025.00092},
   booktitle={2025 IEEE/CVF Winter Conference on Applications of Computer Vision Workshops (WACVW)},
   publisher={IEEE},
   author={Erregue, Iñaki and Nasrollahi, Kamal and Escalera, Sergio},
   year={2025},
   month=feb,
   pages={776-785}
}
```
