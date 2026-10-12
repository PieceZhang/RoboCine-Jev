#!/usr/bin/env python3
"""Package audited demo figures and annotated copies of actual Isaac videos."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np


TITLES = dict(crossing='Crossing post', overtake='Passing cart', two_movers='Two independent movers')


def annotate_video(folder, report):
    import cv2
    original = folder / 'isaac_video/fp_cam.mp4'
    times_file = folder / 'isaac_video/frame_times.json'
    if not original.exists() or not times_file.exists():
        return None
    with (folder / 'telemetry.csv').open() as source:
        rows = list(csv.DictReader(source))
    duration = report['execution_duration_s']
    rows = [row for row in rows if float(row['wall_elapsed_s']) <= duration + .1]
    elapsed = np.array([float(row['wall_elapsed_s']) for row in rows])
    audit = np.load(folder / 'dynamic_audit.npz')
    times = json.loads(times_file.read_text())
    capture = cv2.VideoCapture(str(original))
    fps = capture.get(cv2.CAP_PROP_FPS)
    width, height = [int(capture.get(key)) for key in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT)]
    report_height = height + 166
    target = folder / 'demo_report.mp4'
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, report_height))
    if not capture.isOpened() or not writer.isOpened() or fps <= 0:
        raise RuntimeError(f'cannot read/write report video: {folder}')
    count = 0
    try:
        for frame_time in times:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError('video is shorter than the saved frame timeline')
            t = frame_time['scene_time_s']
            if t > duration + .1:
                break
            i = int(np.clip(np.searchsorted(elapsed, t, side='right') - 1, 0, len(rows) - 1))
            row = rows[i]
            clearance = audit['dynamic_clearance_lower'][i]
            # Keep all native scene pixels visible; captions sit outside them.
            panel = np.full((report_height, width, 3), (24, 31, 40), dtype=np.uint8)
            panel[78:78 + height] = frame
            frame = panel
            def text(value, xy, scale=.62, color=(245, 245, 245)):
                text_width = cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
                scale = min(scale, scale * (width - xy[0] - 20) / max(1, text_width))
                cv2.putText(frame, value, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
            text(f"LiDAR dynamic KC-WBC  |  {TITLES.get(folder.name, folder.name)}", (20, 30), .8)
            static_gap = (audit['static_clearance_lower'][i]
                          if 'static_clearance_lower' in audit.files else None)
            text(f"{report['trajectory']}  |  {Path(report['stage_view']).name if report.get('stage_view') else 'Isaac recording'}"
                 "  |  Robot/mover, robot/furniture, mover/furniture contacts: 0", (20, 60))
            text(f"Physical time {t:5.1f} s    Camera path {float(row['time_s']):5.1f} s"
                 f"    Motion tracks {int(row['dynamic_tracks'])}", (20, report_height - 58))
            gap_text = f"   Dynamic gap {100 * clearance:5.1f} cm"
            if static_gap is not None:
                gap_text += f"   Furniture gap {100 * static_gap:5.1f} cm"
            text(f"Camera error {100 * float(row['position_error']):4.2f} cm" + gap_text
                 + f"   Path rate {float(row['path_rate']):.2f}", (20, report_height - 30))
            progress = np.clip(t / duration, 0., 1.)
            cv2.rectangle(frame, (0, report_height - 6), (int(width * progress), report_height), (104, 198, 58), -1)
            writer.write(frame)
            count += 1
    finally:
        capture.release()
        writer.release()
    if not count:
        raise RuntimeError('no report video frames')
    if shutil.which('ffmpeg') is None:
        raise RuntimeError('ffmpeg is required to export browser-compatible H.264 report videos')
    encoded = target.with_name('demo_report_h264.mp4')
    subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-y', '-i', str(target),
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p',
        '-threads', '4', '-movflags', '+faststart', str(encoded)], check=True)
    encoded.replace(target)
    return dict(path=target.name, frames=count, fps=fps, duration_s=count / fps,
                source_resolution=[width, height], report_resolution=[width, report_height],
                codec='H.264 / yuv420p',
                source='actual Isaac RGB frames; captions from saved telemetry and independent audit')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path)
    parser.add_argument('--figures-only', action='store_true', help='refresh PDF and reuse saved video metadata')
    args = parser.parse_args()
    root = args.runs.expanduser().resolve()
    reports = json.loads((root / 'comparison.json').read_text())
    if len(reports) != 3 or not all(report['dynamic_demo_pass'] for report in reports):
        parser.error('package the three report presets only after every independent audit passes')
    shared_sources = set.intersection(*(set(report['source_sha256']) for report in reports))
    if any(len({report['source_sha256'][source] for report in reports}) != 1 for source in shared_sources):
        parser.error('all three demos must use the same shared runtime sources')
    if len({report.get('stage_view') for report in reports}) != 1:
        parser.error('all three demos must use the same USD scene')
    furnished = all(report.get('static_reference_metadata') for report in reports)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    saved_artifacts = root / 'report_artifacts.json'
    artifacts = json.loads(saved_artifacts.read_text()) if args.figures_only and saved_artifacts.exists() else {}
    with PdfPages(root / 'dynamic_demo_report.pdf') as pdf:
        fig = plt.figure(figsize=(11.7, 8.3))
        fig.text(.07, .9, 'Dynamic LiDAR KC-WBC: three Isaac demos', fontsize=21, weight='bold')
        fig.text(.07, .83, 'Live dual LiDAR perception; unchanged camera trajectories and static trajfollow.', fontsize=11)
        if furnished:
            fig.text(.07, .79, f"Scene: {Path(reports[0]['stage_view']).name}; full apartment visuals and source-layer audit.", fontsize=11)
        ax = fig.add_axes([.07, .43, .86, .32])
        ax.axis('off')
        data = [[TITLES[r['name']], r['trajectory'].replace('shot_', ''),
                 f"{r['position_rms_cm']:.3f}", f"{r['position_peak_cm']:.3f}",
                 f"{100*r['minimum_sampled_dynamic_clearance_lower_m']:.2f}",
                 f"{100*r['minimum_sampled_static_clearance_lower_m']:.2f}" if furnished else 'n/a',
                 f"{100*r['minimum_command_sweep_lower_m']:.2f}", '0 / 0 / 0', 'PASS'] for r in reports]
        table = ax.table(cellText=data, colLabels=['Scene', 'Camera path', 'RMS cm', 'Peak cm',
            'Dynamic cm', 'Furniture cm', 'Sweep cm', 'Contacts', 'Audit'], loc='center', cellLoc='center',
            colWidths=[.18, .15, .09, .09, .10, .10, .10, .12, .07])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2.5)
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor('#e4edf6')
                cell.set_text_props(weight='bold')
        if all('predicted_unmatched_moving_observations' in r for r in reports):
            counts = ', '.join(f"{r['name']}: {r['predicted_unmatched_moving_observations']}" for r in reports)
            fig.text(.07, .43, 'Occlusion forecasts beyond 30 cm: ' + counts + '. Full records retained in audit JSON.', fontsize=9, va='top')
            peak_forecast = max(r.get('predicted_center_error_max_m') or 0. for r in reports)
            fig.text(.07, .407, f'Maximum forecast-centre error: {100*peak_forecast:.2f} cm. Distinct measured-target mismatches: 0.', fontsize=9, va='top')
        fig.text(.07, .365, 'Acceptance: complete trajectory, camera shot checks, live moving tracks, action timing,\n'
            'hard command limits, clearance >= 2.5 cm and zero contacts in all three categories.\n'
            'Camera error statistics cover the shot motion intervals.\n'
            'Truth geometry is used only in the offline audit; runtime sees LiDAR measurements.', fontsize=11, linespacing=1.7, va='top')
        servo_text = ('Physical servo acceleration / lateral-velocity checks passed.'
                      if all(r['physics_constraints_pass'] for r in reports) else
                      'Physical servo acceleration / lateral-velocity checks did not pass in every demo.')
        scope = ('Whole robot and obstacle routes stay inside the audited 4.5 m physics workspace.\n'
                 if furnished else '')
        fig.text(.07, .24, 'Scope: truth clearance at recorded control states; contacts at every 1 ms physics step.\n'
            + scope + 'Command limits passed. ' + servo_text + '\n'
            'Raw telemetry, frozen inputs, source hashes and audit JSON accompany each demo.', fontsize=9, color='#444444', linespacing=1.6, va='top')
        pdf.savefig(fig)
        plt.close(fig)
        for report in reports:
            folder = root / report['name']
            if not args.figures_only:
                artifacts[report['name']] = annotate_video(folder, report)
            fig, ax = plt.subplots(figsize=(11.7, 8.3))
            ax.imshow(plt.imread(folder / 'report.png'))
            ax.axis('off')
            fig.tight_layout(pad=0)
            pdf.savefig(fig)
            plt.close(fig)
    (root / 'report_artifacts.json').write_text(json.dumps(artifacts, indent=2) + '\n')
    print(json.dumps(dict(report=str(root / 'dynamic_demo_report.pdf'), videos=artifacts), indent=2))


if __name__ == '__main__':
    main()
