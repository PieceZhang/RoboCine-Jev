# VLMIntention 障碍物方向标注使用指南

障碍物判定本质是几何问题（能不能撞上），所以**主方案用单目深度模型**（Depth Anything V2 metric）做确定性阈值判定，替代早期不稳定的 VLM 方案。原 637 条 VLM 标注保留为**校准集**，用来标定阈值并评估一致性。

两套标注产物 schema 完全一致（扁平 5 方向布尔），下游验证界面无需改动。

---

## 方案一：深度模型标注（主方案）

### 1. 安装依赖
```bash
cd /Users/siyuwang/Downloads/Project/Project_Robot/VLMIntention_ws
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision transformers pillow     # 首次运行会下载 DA-V2 checkpoint（~1.3GB）
```
> M4 Pro 上 torch 走 MPS 加速。当前 `python3` 已是 arm64，可直接用。

### 2. 运行
```bash
# 测试模式（前 10 张，验证阈值直觉）
python depth_annotation.py

# 全量标注（637 张，支持断点续传）→ 写出 obstacle_annotations_depth.json
python run_depth_annotation.py
```

### 3. 判定逻辑（确定性）
对每张图推理米制深度图，按 5 个区域做阈值判定：
- **左 / 前 / 右** = 图像横向三等分竖条，各占全高
- **上 / 下** = 顶部 / 底部各 25% 行的横条，各占全宽

每区域：`有效深度(0.1~80m) 中 depth < 距离阈值 的像素占比 > 占比阈值` → 该方向有障碍。

> 深度模型不认识"人"，因此**任何近距离表面都算障碍**（含画面主体）。对导航机器人语义更正确。

### 4. 可调常量（`depth_annotation.py` 顶部，校准后回填）
| 常量 | 默认 | 说明 |
|---|---|---|
| `MODEL_ID` | `depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf` | 必须用 `-hf` 后缀的 transformers 格式仓库；室外/街景可换 `-Outdoor-Large-hf` |
| `DIST_THRESH_HORIZ` | 1.5 m | 前/左/右 碰撞距离 |
| `DIST_THRESH_VERT` | 0.5 m | 上/下（沿用原定义） |
| `FRAC_THRESH` | 0.05 | 近距像素占比阈值 |
| `VERT_FRAC` | 0.25 | 上/下 横条高度占比 |

### 5. 输出格式
`dataset/切片/obstacle_annotations_depth.json`：
```json
{
  "分场景切片13-17/17酒店商场广场/标注tag/image.png": {
    "前": true, "左": false, "右": true, "上": false, "下": true,
    "image_name": "image.png",
    "relative_path": "分场景切片13-17/17酒店商场广场/标注tag/image.png",
    "timestamp": "2026-07-20 14:30:00",
    "model": "depth-anything/Depth-Anything-V2-Metric-Hypersim-Large"
  }
}
```

---

## 阈值校准（用 637 条 VLM 标注当参考）

```bash
python calibrate_depth.py              # 一致性报告 + 网格搜索最优阈值（首次推理全图，之后用缓存）
python calibrate_depth.py --no-grid    # 只看当前阈值一致性
python calibrate_depth.py --reinfer    # 强制重新推理刷新缓存
```
- 按方向报 P / R / F1 / 平衡准确率（上 11.8%、下 22.1% 类别稀疏，看 F1 不看裸 accuracy）。
- 网格搜索每方向最优 `DIST_THRESH × FRAC_THRESH`（按 F1），打印推荐值，回填到 `depth_annotation.py`。
- 导出 `depth_calibration_disagreements.json`（与 VLM 分歧最大的样本，便于人工抽查）。

---

## 可视化验证

**逐张看深度图**（原图 / 着色深度图叠 5 区域边界 / 预测标签 + 每区域实测近距占比）：
```bash
.venv/bin/python serve_depth_viewer.py          # 默认 8766，自动开浏览器；← → 翻页
.venv/bin/python serve_depth_viewer.py 9000     # 指定端口
```
首次访问某张时按需推理并缓存到 `dataset/切片/depth_maps/`，之后秒开。深色=远，亮/暖=近；白线划分 前/左/右 三等分与 上/下 横条，可直接核对某方向为何被判定为障碍。

**抽查深度标签**（验证界面，读 5 方向布尔）：
```bash
OBSTACLE_JSON=dataset/切片/obstacle_annotations_depth.json python serve_validation.py 15 42 8765
```

---

## 方案二：VLM 标注（现用作校准集，保留）

早期方案，用 LM Studio 的 Qwen3-VL-30b。已产出 637 条标注，现作为深度方案的校准参考，**不再作为主标注**。

```bash
python run_final_annotation.py   # 全量；需先启动 LM Studio 并加载视觉模型，API http://localhost:1234
```
产物 `dataset/切片/obstacle_annotations_final.json`，schema 与深度版一致。

> VLM 版的"排除画面主体"规则与深度版"近距离即障碍"口径不同；校准时把 VLM 当有噪参考即可，不必强求 100% 一致。

---

## 通用

### 断点续传
两套脚本都自动跳过已标注图片，支持中断后继续。

### 分批处理
```python
annotator.batch_annotate(max_images=100, start_from=0)   # 前 100 张
annotator.batch_annotate(max_images=100, start_from=100) # 101-200
```

### 故障排除
- **MPS 数值异常**：`infer_depth` 已加 finite mask 兜底；仍出错可在 `DepthObstacleAnnotator(base, device="cpu")` 回退 CPU。
- **模型下载慢**：DA-V2 checkpoint 托管在 HuggingFace，必要时设 `HF_ENDPOINT` 镜像。
- **VLM API 连接失败**（方案二）：检查 LM Studio 是否运行、API 是否启用、`http://localhost:1234/v1/models`。
