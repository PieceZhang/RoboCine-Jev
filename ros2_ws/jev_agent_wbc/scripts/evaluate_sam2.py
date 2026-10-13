#!/usr/bin/env python3
"""Offline SAM 2.1 LARGE video masks and fixed-original cinematic metrics.

Run in the user's sam2 conda environment after Isaac has exited. Box seeds only
identify the first-frame subject; every measured mask comes from SAM propagation.
"""
import argparse
import contextlib
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jev_agent_wbc.evaluation import visual_errors, observation_angle, save_json


def segment(predictor, directory, device):
    import torch
    from PIL import Image
    files = sorted(directory.glob('*.jpg'))
    if not files:
        raise ValueError(f'no video frames in {directory}')
    seed = json.loads((directory/'seed.json').read_text())
    state = predictor.init_state(str(directory), offload_video_to_cpu=True, offload_state_to_cpu=True)
    predictor.add_new_points_or_box(inference_state=state, frame_idx=0, obj_id=1,
                                    box=np.asarray(seed['box_xyxy'], dtype=np.float32))
    h, w = state['video_height'], state['video_width']
    masks = np.zeros((len(files), h, w), dtype=bool)
    statuses = ['missing_prediction']*len(files)
    for frame, ids, logits in predictor.propagate_in_video(state):
        if ids != [1] or tuple(logits.shape) != (1, 1, h, w) or not torch.isfinite(logits).all():
            statuses[frame] = 'invalid_logits'
            continue
        masks[frame] = (logits[0, 0] > 0.).detach().cpu().numpy()
        statuses[frame] = 'valid' if masks[frame].any() else 'empty_mask'
    np.savez_compressed(directory.parent/(directory.name+'_sam21_masks.npz'), masks=masks, status=np.array(statuses))
    # Small overlays make segmentation failure visible to a reviewer.
    for i in np.linspace(0, len(files)-1, min(6, len(files)), dtype=int):
        rgb = np.array(Image.open(files[i]).convert('RGB'))
        overlay = rgb.copy()
        overlay[masks[i]] = (.45*rgb[masks[i]]+.55*np.array([30, 245, 70])).astype(np.uint8)
        Image.fromarray(overlay).save(directory.parent/f'{directory.name}_mask_{i:05d}.png')
    predictor.reset_state(state)
    del state
    return masks, statuses


def evaluate(predictor, folder, model_info):
    video = folder/'video'
    frames = json.loads((video/'frames.json').read_text())
    reference, reference_status = segment(predictor, video/'reference', model_info['device'])
    actual, actual_status = segment(predictor, video/'execution', model_info['device'])
    if len(reference) != len(actual) or len(frames) != len(actual):
        raise ValueError('frame correspondence mismatch; refusing to retime evaluation')
    values = []
    task = json.loads((folder/'original_task.json').read_text())
    for i, f in enumerate(frames):
        row = visual_errors(reference[i] if reference_status[i] == 'valid' else None,
                            actual[i] if actual_status[i] == 'valid' else None)
        row.update(frame=i, time_s=f['t'], reference_segmentation_status=reference_status[i],
                   phase=next((p['name'] for p in task['shot_phase'] if f['t'] < p['end_s']), task['shot_phase'][-1]['name']),
                   execution_segmentation_status=actual_status[i],
                   observation_angle_error_rad=observation_angle(f['reference_camera_position'],
                       f['actual_camera_position'], f.get('subject_track') or {}),
                   visibility=f.get('visibility', {'available': False}))
        values.append(row)
    def stats(key, rows=None):
        rows = values if rows is None else rows
        vals = [row[key] for row in rows if row.get(key) is not None]
        return dict(valid_frames=len(vals), invalid_frames=len(rows)-len(vals),
                    mean=None if not vals else float(np.mean(vals)),
                    rms=None if not vals else float(np.sqrt(np.mean(np.asarray(vals)**2))),
                    max=None if not vals else float(np.max(vals)))
    result = dict(model=model_info, correspondence='original_elapsed_time; no time warping', frames=len(values),
        composition_error=stats('composition_error'), subject_scale_error=stats('subject_scale_error'),
        observation_angle_error_rad=stats('observation_angle_error_rad'),
        segmentation_failure_frames=sum(not row['valid'] for row in values),
        execution_frame_edge_masks=sum(row['valid'] and row['execution']['frame_edge'] for row in values),
        reference_frame_edge_masks=sum(row['valid'] and row['reference']['frame_edge'] for row in values),
        occluded_frames=sum(row['visibility'].get('occluded') is True for row in values),
        out_of_frame_frames=sum(row['visibility'].get('out_of_frame') is True for row in values),
        unknown_visibility_frames=sum(not row['visibility'].get('available', False) for row in values),
        mask_source='SAM 2.1 video predictor; simulator geometry only seeds the first-frame subject box',
        angle_source='timestamped LiDAR subject track and measured camera pose; invalid tracks excluded and counted',
        per_frame=values)
    # Data-quality screening only: flag abrupt mask changes for visual review.
    # Do not silently remove frames or alter the fixed error definitions.
    jumps = {}
    for stream in ('reference', 'execution'):
        areas = [None if not row.get(stream) else row[stream]['area_scale'] for row in values]
        jumps[stream] = [i for i in range(1, len(areas)) if areas[i] and areas[i-1]
                        and not .25 <= areas[i]/areas[i-1] <= 4.]
    rendering = video/'reference_rendering.json'
    rendering_info = (json.loads(rendering.read_text()) if rendering.exists() else
        dict(protocol='legacy_live_visibility_toggle', independently_checked=False))
    result['quality_control'] = dict(mask_area_jump_frames=jumps,
        ratio_screen=[.25, 4.], automatically_excluded=False,
        review_required=bool(any(jumps.values()) or result['segmentation_failure_frames']
                             or rendering_info.get('valid') is not True),
        reference_rendering=rendering_info)
    result['phases'] = {p['name']: {key: stats(key, [row for row in values if row['phase'] == p['name']])
        for key in ('composition_error', 'subject_scale_error', 'observation_angle_error_rad')}
        for p in task['shot_phase']}
    save_json(folder/'visual_metrics.json', result)
    summary = json.loads((folder/'summary.json').read_text())
    summary['sam_visual_measurement'] = 'completed' if not result['segmentation_failure_frames'] else 'completed_with_invalid_frames'
    summary['visual_metrics'] = {k:v for k,v in result.items() if k != 'per_frame'}
    summary['observation_angle_measurement'] = 'completed' if result['observation_angle_error_rad']['valid_frames'] else 'unavailable_lidar_track'
    save_json(folder/'summary.json', summary)
    print(json.dumps(dict(folder=str(folder), frames=len(values), invalid=result['segmentation_failure_frames'],
        composition=result['composition_error']['mean'], scale=result['subject_scale_error']['mean'])), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True, help='case folder or parent suite folder')
    parser.add_argument('--sam-root', type=Path, default=ROOT.parents[1]/'sam21')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args()
    checkpoint = args.checkpoint or args.sam_root/'checkpoints/sam2.1_hiera_large.pt'
    if not checkpoint.is_file():
        parser.error(f'large SAM 2.1 checkpoint missing: {checkpoint}')
    import torch
    from sam2.build_sam import build_sam2_video_predictor
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in sam2 environment')
    model_info = dict(variant='sam2.1_hiera_large', config='configs/sam2.1/sam2.1_hiera_l.yaml',
                      checkpoint=str(checkpoint.resolve()), device=args.device,
                      checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    paths = [args.runs] if (args.runs/'video/frames.json').exists() else sorted(p.parent.parent for p in args.runs.glob('*/video/frames.json'))
    if not paths:
        parser.error('no completed RGB pairs with frames.json')
    context = torch.autocast('cuda', dtype=torch.bfloat16) if args.device == 'cuda' else contextlib.nullcontext()
    with torch.inference_mode(), context:
        predictor = build_sam2_video_predictor(model_info['config'], str(checkpoint), device=args.device)
        for path in paths:
            evaluate(predictor, path, model_info)
            gc.collect()
            if args.device == 'cuda':
                torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
