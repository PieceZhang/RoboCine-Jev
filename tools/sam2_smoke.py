#!/usr/bin/env python3
"""Check both SAM 2.1 checkpoints with real image and short-video inference."""

import argparse
import gc
import json
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
from PIL import Image
import torch

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.utils.misc import get_connected_components


ROOT = Path(__file__).resolve().parents[1] / "sam21"
MODELS = {
    "base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "large": "configs/sam2.1/sam2.1_hiera_l.yaml",
}


def check_image(variant, config, checkpoint):
    image = np.array(Image.open(ROOT / "notebooks/images/truck.jpg").convert("RGB"))
    predictor = SAM2ImagePredictor(build_sam2(config, str(checkpoint)))
    torch.cuda.synchronize()
    start = time.perf_counter()
    predictor.set_image(image)
    masks, scores, logits = predictor.predict(
        point_coords=np.array([[500, 375]], dtype=np.float32),
        point_labels=np.array([1], dtype=np.int32),
        multimask_output=True,
    )
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000
    if masks.shape != (3, *image.shape[:2]) or not masks.any():
        raise RuntimeError(f"{variant}: invalid image masks {masks.shape}")
    if not np.isfinite(scores).all() or not np.isfinite(logits).all():
        raise RuntimeError(f"{variant}: non-finite image prediction")
    return {"mask_shape": list(masks.shape), "scores": scores.tolist(),
            "image_ms": round(elapsed_ms, 2)}


def check_video(config, checkpoint):
    predictor = build_sam2_video_predictor(config, str(checkpoint))
    with tempfile.TemporaryDirectory(prefix="sam21-smoke-") as directory:
        for index in range(3):
            source = ROOT / f"notebooks/videos/bedroom/{index:05d}.jpg"
            shutil.copyfile(source, Path(directory) / source.name)
        state = predictor.init_state(directory, offload_video_to_cpu=True)
        height, width = state["video_height"], state["video_width"]
        predictor.add_new_points_or_box(
            inference_state=state, frame_idx=0, obj_id=1,
            points=np.array([[210, 350]], dtype=np.float32),
            labels=np.array([1], dtype=np.int32),
        )
        frames = []
        torch.cuda.synchronize()
        start = time.perf_counter()
        for frame, ids, masks in predictor.propagate_in_video(state):
            if ids != [1] or tuple(masks.shape) != (1, 1, height, width):
                raise RuntimeError(f"Invalid video output on frame {frame}")
            if not torch.isfinite(masks).all() or not (masks > 0).any():
                raise RuntimeError(f"Invalid video mask on frame {frame}")
            frames.append(frame)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000
        if frames != [0, 1, 2]:
            raise RuntimeError(f"Missing video frames: {frames}")
    return {"video_frames": len(frames), "video_total_ms": round(elapsed_ms, 2)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["both", *MODELS], default="both")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this check on the NVIDIA GPU")
    print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda,
                      "gpu": torch.cuda.get_device_name(0)}, ensure_ascii=False), flush=True)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        binary = torch.zeros((1, 1, 8, 8), dtype=torch.uint8, device="cuda")
        binary[:, :, 2:5, 2:5] = 1
        labels, areas = get_connected_components(binary)
        if not (labels[:, :, 2:5, 2:5] > 0).all() or not (areas[:, :, 2:5, 2:5] == 9).all():
            raise RuntimeError("CUDA connected-components extension failed")
        print("CUDA post-processing extension: OK", flush=True)
        for variant, config in MODELS.items():
            if args.model != "both" and args.model != variant:
                continue
            checkpoint = ROOT / f"checkpoints/sam2.1_hiera_{variant}.pt"
            result = {"model": variant, **check_image(variant, config, checkpoint)}
            gc.collect()
            torch.cuda.empty_cache()
            result.update(check_video(config, checkpoint))
            print(json.dumps(result, ensure_ascii=False), flush=True)
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
