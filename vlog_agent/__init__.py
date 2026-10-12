"""vlog_agent — 素材库 → 分类索引 → 叙事 EDL → ffmpeg 成片。

三阶段（run_vlog.py --stage {ingest,storyboard,edit,all}）:
  ingest     扫描素材树（mp4/mov/jpg + DJI LRF/SRT 边车）→ ffprobe meta +
             代理抽帧 + VLM 6 维分类 → material_index.json
  storyboard LLM 读索引 → 选段/排序/字幕 → edl.json（校验失败回退规则选段）
  edit       按 EDL 用 ffmpeg 归一化切片 + 拼接 → edits/<ts>/vlog.mp4

模型走 LM Studio 的 OpenAI 兼容端点（与 cinema_traj.plan、
vlm_intention.annotate_vlm 同一接入方式）。
"""
