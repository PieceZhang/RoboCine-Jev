# vlog_agent — 素材库 → 分类索引 → 叙事 EDL → 成片

基于拍摄素材（mp4/mov/jpg + DJI 边车文件）的自动 vlog 剪辑 agent，
三阶段由 `run_vlog.py` 编排，模型走 LM Studio（与 `cinema_traj.plan`、
`vlm_intention.annotate_vlm` 同一 OpenAI 兼容端点）。

```
/Volumes/... 素材树
   │ ① ingest     扫描 → ffprobe meta → 代理抽帧 → VLM 分类
   ▼              → vlog_agent_out/material_index.json（逐文件断点续跑）
   │ ② storyboard LLM 读索引 + 主题 → 选段/排序/字幕
   ▼              → vlog_agent_out/edl.json（校验失败自动回退规则选段）
   │ ③ edit       ffmpeg 归一化切片（字幕用 PIL PNG + overlay）→ concat
   ▼              → vlog_agent_out/edits/<ts>/vlog.mp4 + edit_report.json
```

## 用法

```bash
source .venv/bin/activate
# 分步
python run_vlog.py --stage ingest     --src /path/to/素材 [--force] [--dry-vlm]
python run_vlog.py --stage storyboard --brief "风电场自驾旅拍" --target 30 [--photos]
python run_vlog.py --stage edit
# 一次跑完
python run_vlog.py --stage all --src /path/to/素材 --brief "..." --target 30
# 模型：--model 显式指定；缺省自动探测 LM Studio 第一个非 embedding 模型
```

2026-09-08 用 LM Studio `qwen3-vl-2b-instruct` 端到端闭环验证
（`vlog_agent_out/smoke_src/`：4 iPhone 短片 + 2 DJI Mini4 Pro 视频 + 1 DJI 照片）：
ingest 7/7 分类成功（场景/主体/运镜/质量/问题/描述），storyboard 出 6 段合法 EDL，
edit 6/6 渲染 + concat 成 13.1s 1920x1080 成片。

## 素材索引条目（material_index.json）

```json
{"id": "相对路径", "type": "video|photo", "path": "绝对路径",
 "proxy": "同stem .LRF 代理", "telemetry": "同stem .SRT",
 "duration": 秒, "width": 1920, "height": 1080, "fps": 24.0,
 "device": "DJI Mini4 Pro（encoder tag）", "created_at": "ISO 时间",
 "gps": "SRT 首个 GPS 定位（原始格式）", "dir_hint": "父目录名",
 "frames": ["抽帧缩略图绝对路径"],
 "classified": true, "model": "qwen3-vl-2b-instruct",
 "scene": "室内|室外人工|室外自然", "subject": "单人|多人|空镜|物体",
 "shot": "航拍|地面|手持|其他", "quality": 1-5,
 "issues": ["抖动|过曝|欠曝|模糊|无"], "desc": "≤25字中文描述", "error": null}
```

分类词表是**硬校验**（`sanitize_tags` 把 VLM 自由输出钳回固定词表），
与 `README_VLMLableInstruction.md` 的 6 维标注体系对齐（场景大类/
主体大类/运镜小类；小类细分留给后续接入上游模型）。

## 设计要点

- **LRF 代理优先**：抽帧只读同 stem 的 .LRF（DJI 低码率代理），剪辑才碰
  原片——机械硬盘上全片扫描不可用。
- **幂等**：索引按 id 增量写（每文件一个 checkpoint，tmp+rename 原子替换），
  重跑跳过已分类条目；`--force` 才重分类。
- **EDL 永不落空**：LLM 输出经 `validate_edl` 钳制（id 存在性/时长边界/
  去重/照片固定 3s），两轮仍不合法则回退 `rule_based_edl`
  （质量优先 + 场景多样性 + 按时间排序）。
- **字幕不依赖 ffmpeg 文字能力**：PIL 渲染透明 PNG（PingFang）+ overlay 滤镜
  ——本机 ffmpeg 9.0.1 无 freetype（无 drawtext/subtitles），overlay 哪都有。
- **成片归一化**：所有片段统一 H.264 1920x1080 30fps + AAC 48k 立体声后
  concat 流拷贝（失败自动降级重编码）；原声保留，无声素材（照片）补静音轨。

## 已知限制（v1）

- 2B 模型给出的 EDL 偏保守（目标 30s 实际 ~13s）；换 30B-A3B 后重评。
- 照片段固定 3s 静音 + 慢推近，无转场特效（cut-only）；配乐未接。
- HYPERLAPSE/PANORAMA/HEIC/DNG/Photos Library 默认排除（`.photoslibrary`、
  帧序列目录）；HEIC 需要时再接 Pillow-HEIF。
- concat 流拷贝产生轻微 VFR（~29.8fps 均值），播放器无感；要严格 CFR 可
  在 `_concat` 里强制重编码。
- `.video_agent` 的 DJI 无人机素材尚未接 GPS → 场景聚类（SRT 已采集，未利用）。
