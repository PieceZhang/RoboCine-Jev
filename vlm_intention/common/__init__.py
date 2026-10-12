"""Shared, deduplicated logic for the VLMIntention pipeline.

Submodules (import explicitly — depth_model pulls torch/transformers):
  config      — YAML loader + ${paths.base} interpolation
  schema      — DIRS + the 5-bool annotation envelope
  thresholds  — DIST_THRESH / FRAC_THRESH / DEPTH_MIN-MAX / CONFIG_SIG
  atomic_io   — atomic_write_json / backup_once / load_json
  paths       — resolve_base / mp4_for / mp4_rel_for / safe_resolve / discover_*
  env         — indoor/outdoor classification
  geometry    — 5-region depth stats / score_frame / close_frac_from_hist
  gt          — make_save_gt(snapshot_key) writeback factory
  depth_model — DepthObstacleAnnotator (lazy HF cache, infer_depth, batch_annotate)
"""
