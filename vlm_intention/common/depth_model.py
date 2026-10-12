"""DepthObstacleAnnotator — monocular depth-model obstacle-direction labeler.

Runs Depth-Anything-V2 (metric) per image, slices into 5 regions, applies
per-direction thresholds to emit the 5-bool schema. Relocated from
depth_annotation.py; now config-driven and reusing common.geometry/env/atomic_io.

Behavior preserves the pre-refactor output byte-for-byte when config defaults
are in effect (Indoor checkpoint for both envs, per-direction thresholds).
"""
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from . import env as envmod
from . import geometry
from .atomic_io import atomic_write_json, load_json, backup_once
from .paths import discover_images
from .schema import DIRS, build_annotation_record
from .thresholds import (
    DIST_THRESH, FRAC_THRESH, VERT_FRAC, DEPTH_MIN, DEPTH_MAX,
    CONFIG_VERSION, build_config_sig,
)

MODEL_ID_INDOOR = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
MODEL_ID_OUTDOOR = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"


class DepthObstacleAnnotator:
    def __init__(self, base_path, device: Optional[str] = None, force: bool = False,
                 read_only: bool = False, cfg=None):
        self.base_path = Path(base_path)
        self.cfg = cfg
        self.force = force
        self.read_only = read_only

        # --- config-driven thresholds/ids (defaults reproduce old behavior) ---
        th = (cfg.thresholds if cfg else None)
        self.dist_thresh = (th.dist_thresh if th and "dist_thresh" in th else DIST_THRESH)
        self.frac_thresh = (th.frac_thresh if th and "frac_thresh" in th else FRAC_THRESH)
        self.vert_frac = (th.vert_frac if th and "vert_frac" in th else VERT_FRAC)
        self.depth_min = (th.depth_min if th and "depth_min" in th else DEPTH_MIN)
        self.depth_max = (th.depth_max if th and "depth_max" in th else DEPTH_MAX)
        self.save_every = (cfg.sampling.save_every if cfg and "sampling" in cfg else 10)

        md = (cfg.models.depth if cfg and "models" in cfg else None)
        self.model_indoor = (md.indoor if md and "indoor" in md else MODEL_ID_INDOOR)
        self.model_outdoor = (md.outdoor if md and "outdoor" in md else MODEL_ID_OUTDOOR)
        alias = (md.outdoor_alias if md and "outdoor_alias" in md else "indoor")
        self.env_model_id = {
            "indoor": self.model_indoor,
            "outdoor": self.model_indoor if alias == "indoor" else self.model_outdoor,
        }
        self.config_sig = build_config_sig(
            self.dist_thresh, self.frac_thresh, self.vert_frac, CONFIG_VERSION)

        # --- device ---
        if device:
            self.device = torch.device(device)
        else:
            devpref = (md.device if md and "device" in md else "auto")
            if devpref in ("mps", "cpu"):
                self.device = torch.device(devpref)
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")

        # --- output paths ---
        self.annotation_file = self.base_path / (
            cfg.outputs.depth_annotations if cfg and "outputs" in cfg
            else "obstacle_annotations_depth.json")
        self.meta_file = self.base_path / (
            cfg.outputs.depth_meta if cfg and "outputs" in cfg
            else "obstacle_annotations_depth.meta.json")

        self.load_existing_annotations()
        self.existing_meta = self._load_meta_entries()
        self._models = {}                       # lazy: model_id -> (proc, model, id)
        self._scene_env = envmod.load_scene_env(self.base_path)
        self.keywords = (cfg.filter.indoor_keywords if cfg and "filter" in cfg
                         and "indoor_keywords" in cfg.filter else envmod.INDOOR_KEYWORDS)
        self.default_env = (cfg.filter.default_env if cfg and "filter" in cfg
                            and "default_env" in cfg.filter else "outdoor")

        self.stale = self._detect_stale()
        if not read_only:
            if self.stale:
                print("⚠️  深度标注配置已变更（阈值/模型方案），现有结果为旧配置产物。")
                print("   建议用 --force 全量重算。")
            if self.force:
                backup_once(self.annotation_file)
                self.existing_annotations = {}
                self.existing_meta = {}
                print("🔄 --force：全量重算（旧主结果已备份）。")

    # ── io ──
    def load_existing_annotations(self):
        self.existing_annotations = load_json(self.annotation_file, default={})

    def _load_meta(self):
        m = load_json(self.meta_file, default=None)
        if isinstance(m, dict) and "_config" in m:
            return m
        return None

    def _load_meta_entries(self):
        m = self._load_meta()
        return {k: v for k, v in (m or {}).items() if k != "_config"} if m else {}

    def _detect_stale(self) -> bool:
        meta_disk = self._load_meta()
        if meta_disk is None:
            return (not self.read_only) and bool(self.existing_annotations)
        return meta_disk.get("_config", {}).get("sig") != self.config_sig

    def save_annotations(self):
        atomic_write_json(self.annotation_file, self.existing_annotations)

    def save_meta(self):
        out = dict(self.existing_meta)
        out["_config"] = {
            "sig": self.config_sig, "version": CONFIG_VERSION,
            "thresholds": self.dist_thresh, "frac_thresh": self.frac_thresh,
            "vert_frac": self.vert_frac,
            "indoor_model": self.model_indoor, "outdoor_model": self.model_outdoor,
            "depth_min": self.depth_min, "depth_max": self.depth_max,
        }
        atomic_write_json(self.meta_file, out)

    # ── discovery ──
    def get_all_images(self):
        return discover_images(self.base_path)

    def get_images_per_category(self, per_cat, seed=0):
        """Group by direct parent dir; take per_cat per group (seeded)."""
        import random
        groups = {}
        for p in self.get_all_images():
            groups.setdefault(p.parent, []).append(p)
        rng = random.Random(seed)
        picked = []
        for parent in sorted(groups):
            items = sorted(groups[parent])
            rng.shuffle(items)
            picked.extend(items[:per_cat])
        return sorted(picked)

    # ── env ──
    def classify_env(self, image_path):
        return envmod.classify_env(image_path, self._scene_env,
                                   keywords=self.keywords, default=self.default_env)

    # ── depth ──
    def _ensure_model(self, env: str):
        mid = self.env_model_id.get(env, self.model_indoor)
        if mid not in self._models:
            print(f"🤖 加载深度模型[{env}]: {mid}")
            proc = AutoImageProcessor.from_pretrained(mid)
            mdl = AutoModelForDepthEstimation.from_pretrained(mid).to(self.device).eval()
            self._models[mid] = (proc, mdl, mid)
        return self._models[mid]

    def infer_depth(self, image_path, env) -> Optional[np.ndarray]:
        try:
            proc, mdl, _ = self._ensure_model(env)
            image = Image.open(image_path).convert("RGB")
            orig_w, orig_h = image.size
            inputs = proc(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = mdl(**inputs)
            depth = outputs.predicted_depth
            if depth.dim() == 2:
                depth = depth.unsqueeze(0).unsqueeze(0)
            elif depth.dim() == 3:
                depth = depth.unsqueeze(0)
            depth = F.interpolate(depth, size=(orig_h, orig_w),
                                  mode="bilinear", align_corners=False)
            return depth[0, 0].cpu().numpy()
        except Exception as e:
            print(f"  - 深度推理失败 {Path(image_path).name}: {e}")
            return None

    # ── region geometry (delegate to common.geometry with this instance's thresholds) ──
    def region_depths(self, depth):
        return geometry.region_depths(depth, vert_frac=self.vert_frac)

    def region_stats(self, depth):
        return geometry.region_stats(
            depth, self.dist_thresh, self.frac_thresh, self.vert_frac,
            self.depth_min, self.depth_max)

    def depth_to_obstacles(self, depth):
        return {d: s["obstacle"] for d, s in self.region_stats(depth).items()}

    # ── single / batch ──
    def annotate_single_image(self, image_path):
        import time
        relative_path = str(Path(image_path).relative_to(self.base_path))
        if relative_path in self.existing_annotations and not self.force:
            return None
        env = self.classify_env(image_path)
        depth = self.infer_depth(image_path, env)
        if depth is None:
            return None
        _, _, model_id = self._ensure_model(env)
        stats = self.region_stats(depth)
        directions = {d: bool(stats[d]["obstacle"]) for d in DIRS}
        annotation = build_annotation_record(
            directions, Path(image_path).name, relative_path, model_id)
        meta = {
            "image_name": Path(image_path).name, "relative_path": relative_path,
            "indoor": env == "indoor", "env": env, "model": model_id,
            "timestamp": annotation["timestamp"], "regions": stats,
        }
        return annotation, meta, env

    def batch_annotate(self, max_images=None, start_from=0, images=None):
        import time
        if images is None:
            images = self.get_all_images()
            print(f"📁 找到 {len(images)} 张图片")
        else:
            print(f"📁 使用传入清单：{len(images)} 张图片")
        if max_images:
            images = images[start_from:start_from + max_images]
        else:
            images = images[start_from:]
        print(f"🎯 将处理 {len(images)} 张  |  阈值: "
              + "  ".join(f"{d}<{self.dist_thresh[d]}m" for d in DIRS)
              + f"  |  占比>{self.frac_thresh}")

        success = error = skip = 0
        indoor_n = outdoor_n = 0
        t0 = time.time()
        for i, image_path in enumerate(images):
            if (i + 1) % 10 == 0:
                print(f"进度: [{i + 1}/{len(images)}] | 已耗时: {time.time()-t0:.1f}秒")
            try:
                result = self.annotate_single_image(image_path)
                if result is None:
                    skip += 1
                    continue
                annotation, meta, env = result
                rel = annotation["relative_path"]
                self.existing_annotations[rel] = annotation
                self.existing_meta[rel] = meta
                success += 1
                indoor_n += env == "indoor"
                outdoor_n += env != "indoor"
                active = [d for d in DIRS if annotation[d]]
                print(f"  ✓ [{env:7s}] {Path(image_path).name[:30]}  障碍: {active if active else '无'}")
                if success % self.save_every == 0:
                    self.save_annotations()
                    self.save_meta()
            except Exception as e:
                print(f"  ❌ 失败 {Path(image_path).name}: {e}")
                error += 1
        self.save_annotations()
        self.save_meta()
        print("=" * 70)
        print(f"🎉 完成！✅ 成功: {success} (室内 {indoor_n}/室外 {outdoor_n}) | "
              f"⏭️ 跳过: {skip} | ❌ 失败: {error}")
        print(f"📊 总计: {len(self.existing_annotations)} 条 | ⏱️ 耗时: {(time.time()-t0)/60:.1f} 分钟")
        return self.existing_annotations

    def get_statistics(self):
        if not self.existing_annotations:
            print("📊 暂无标注数据")
            return
        n = len(self.existing_annotations)
        print(f"\n📊 标注统计（共 {n} 条）:")
        for d in DIRS:
            c = sum(1 for a in self.existing_annotations.values() if a.get(d))
            print(f"  {d} (<{self.dist_thresh[d]}m): {c} ({c/n*100:.1f}%)")
