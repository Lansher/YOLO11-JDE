# 项目脚本说明

可执行脚本集中在 `scripts/`。请在**仓库根目录**下运行，例如：

```bash
python scripts/train.py
python scripts/track.py
```

`bootstrap.py` 会在需要时将仓库根目录加入 `sys.path`，以便导入顶层包 `tracker` 等。

## 脚本列表

| 文件 | 说明 |
|------|------|
| `bootstrap.py` | 将仓库根目录加入 `sys.path`，供需要 `import tracker` 的脚本使用 |
| `train.py` | JDE 主训练流程（Comet、MOT 回调等） |
| `train_optimized.py` | 带优化超参的训练 |
| `train_with_tracks.py` | 使用含 track ID 标注的数据集训练 |
| `track.py` | 视频/序列跟踪示例 |
| `validate.py` / `validate_cpu.py` | 验证（GPU / CPU） |
| `debug.py` | 小规模调试训练 |
| `convert_mot20_gt_to_yolo.py` | MOT20 GT → YOLO 格式（含 track ID） |
| `offline_train.py` | 后台启动 `scripts/train.py`，日志写入 `logs/` |
| `benchmark_deploy.py` | 部署/基准相关脚本 |
| `heatmap_video.py` | 轨迹累积热力视频 |
| `gradcam_video.py` / `gradcam_image.py` | Grad-CAM 可视化 |

## 运行方式

在**仓库根目录**下：

```bash
python scripts/train.py
# 或
cd scripts && python train.py   # 亦可；bootstrap 会处理路径
```

直接运行 `scripts/` 内脚本时，`bootstrap` 会保证能导入顶层包 `tracker`。
