# YOLO11s-JDE 模型量化与推理加速说明

## 1. 文档目的

本文说明如何将权重文件（例如 `YOLO11s_JDE-CHMOT17-64b-100e_TBHS_m075_1280px.pt`）从 PyTorch `.pt` 部署为 **ONNX / TensorRT** 等格式，并通过 **FP16** 或 **INT8 量化** 降低延迟、提高 FPS。

典型加速路径：**PyTorch FP32 → ONNX FP32 → TensorRT FP32 → TensorRT FP16**；在需要进一步压缩算力时，可尝试 **TensorRT INT8**（需校准数据集）。

> **注意**：JDE 模型除检测外还输出 **ReID 嵌入**。INT8 对嵌入区分度的影响通常大于 FP16，上线前务必用 **跟踪指标（如 MOTA/IDF1）** 与业务效果验证，而不仅看检测精度或 FPS。

---

## 2. 环境与依赖

| 项目 | 说明 |
|------|------|
| Python | 与本 YOLO11-JDE 工程一致 |
| PyTorch + CUDA | TensorRT 导出与 FP16/INT8 引擎需在 **NVIDIA GPU** 上完成 |
| TensorRT | 与 CUDA 版本匹配的 TensorRT |
| 导出依赖 | `pip install "ultralytics[export]"`，按报错补装 `onnx`、`onnxruntime-gpu` 等 |

在本仓库开发时，确保使用的 `ultralytics` 与当前代码一致（或从项目根目录以可编辑方式安装）。

---

## 3. 推荐实施顺序

1. **基线**：`.pt` FP32 推理，记录延迟与 FPS。
2. **优先**：导出 **TensorRT + FP16**（实现简单，精度通常可接受）。
3. **可选**：**TensorRT + INT8**（需 `data=xxx.yaml` 校准数据，且需回归跟踪效果）。
4. **中间验证**：导出 **ONNX**，用 ONNX Runtime GPU 或再转 TensorRT。

---

## 4. 任务类型与输入尺寸

- **任务**：加载时必须指定 **`task="jde"`**，否则可能按普通检测加载。
- **分辨率**：文件名中含 `1280px` 时，导出与推理的 **`imgsz` 应与训练一致**（如 `1280` 或 `[H, W]`）。不一致会改变特征尺度，影响速度与精度。

```python
from ultralytics import YOLO

model = YOLO("YOLO11s_JDE-CHMOT17-64b-100e_TBHS_m075_1280px.pt", task="jde")
```

---

## 5. 方案 A：TensorRT FP16（推荐首选）

**适用**：NVIDIA GPU，追求高 FPS / 低延迟。

### 5.1 Python API

```python
from ultralytics import YOLO

model = YOLO("YOLO11s_JDE-CHMOT17-64b-100e_TBHS_m075_1280px.pt", task="jde")
model.export(
    format="engine",   # TensorRT
    device=0,          # GPU 编号
    half=True,         # FP16
    imgsz=1280,        # 与训练一致，或写成 [h, w]
    batch=1,           # 按实际部署 batch 设置
)
```

### 5.2 命令行

```bash
yolo mode=export model=YOLO11s_JDE-CHMOT17-64b-100e_TBHS_m075_1280px.pt \
  format=engine device=0 half=True imgsz=1280 batch=1 task=jde
```

导出得到 `.engine` 文件；推理时加载该引擎并同样指定 `task="jde"`（以当前 ultralytics 版本文档为准）。

---

## 6. 方案 B：TensorRT INT8

**适用**：FP16 仍不满足时，且可接受一定精度与关联质量风险。

**要求**：

- 在 GPU 上导出 `format=engine`。
- 提供 **`data=数据集 yaml`** 用于 INT8 **校准**（建议使用与目标场景分布相近的图像，如 MOT 子集）。
- 框架侧对 TensorRT INT8 可能强制 **`dynamic=True`** 等约束，以导出日志为准；若提示 batch / max batch，按说明设置。

### Python 示例

```python
from ultralytics import YOLO

model = YOLO("YOLO11s_JDE-CHMOT17-64b-100e_TBHS_m075_1280px.pt", task="jde")
model.export(
    format="engine",
    device=0,
    int8=True,
    data="你的数据.yaml",   # 校准用
    imgsz=1280,
    batch=1,
)
```

### JDE 特别说明

- 输出包含 **框 + 类别 + ReID 特征**；INT8 可能导致 **ID 切换增多** 或相似目标混淆。
- 务必用 **跟踪指标** 与 FP16 对比后再上线。

---

## 7. 方案 C：ONNX

便于跨框架验证或多平台部署。

```python
model.export(format="onnx", imgsz=1280, simplify=True, device=0)
```

后续可用 ONNX Runtime 或再转换为 TensorRT。

---

## 8. 推理与跟踪

仓库内可参考 `track.py`：使用 `YOLO(..., task="jde")` 与 `tracker="jdetracker.yaml"`。更换为 `.engine` 后，**`imgsz`、`conf`、`iou` 等** 应与评估时保持一致，便于公平对比速度。

---

## 9. 验收清单

| 项目 | 内容 |
|------|------|
| 正确性 | 同一段视频上，FP32 / FP16 / INT8 的检测与 ID 是否可接受 |
| 速度 | 固定分辨率与 batch，预热后统计平均延迟与 FPS |
| 嵌入质量 | INT8 若跟踪明显变差，优先调整校准集或退回 FP16 |
| 环境 | CUDA、TensorRT、驱动版本固定，避免对比失真 |

---

## 10. 常见问题

| 现象 | 可能原因 | 处理方向 |
|------|----------|----------|
| 导出失败 | 未装 TensorRT 或 CUDA 不匹配 | 对齐版本或使用官方 GPU 容器 |
| INT8 缺少校准数据 | 未指定 `data` | 提供 `data=xxx.yaml` |
| 速度提升不明显 | 预处理/后处理或 IO 占主导 | 拆分测纯推理耗时；优化读帧与后处理 |
| 跟踪变差 | 量化损伤 ReID | 优先 FP16；INT8 用场景数据校准并回归指标 |

---

## 11. 实现参考（本仓库）

- 导出逻辑：`ultralytics/engine/exporter.py`（TensorRT FP16/INT8、`int8` 与 `data` 等）。
- JDE 检测头推理与导出输出：`ultralytics/nn/modules/head.py` 中 `JDE` 类。

---

## 12. 小结

- **优先采用 TensorRT + FP16**，在多数检测+嵌入场景下性价比较高。
- **INT8** 需校准数据，且对 JDE 跟踪效果影响需单独验证。
- 全程保持 **`task="jde"`** 与 **`imgsz` 与训练一致**。

文档版本：随本仓库 Ultralytics 导出行为整理；具体参数以 `ultralytics/cfg/default.yaml` 与 `yolo export` 帮助为准。
