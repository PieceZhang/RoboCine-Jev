"""Camera output contract shared by PhysX, Agent feedback and video evaluation."""
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.spatial.transform import Rotation

from .virtual_stabilizer import StabilizerConfig, VirtualStabilizer


DEFAULT_CAMERA_OUTPUT = 'stabilized'
CAMERA_VIEWS = ('reference', 'execution', 'stabilized')


def add_camera_arguments(parser):
    parser.add_argument('--publish-camera-ros', action='store_true',
                        help='publish the selected camera to /camera/image and its acquisition-time pose/TF')
    parser.add_argument('--camera-output', choices=('stabilized', 'execution'),
                        default=DEFAULT_CAMERA_OUTPUT)
    parser.add_argument('--stabilizer-config', type=Path,
                        help='optional StabilizerConfig JSON; default is the validated 0.5 Hz configuration')


class CameraOutput:
    def __init__(self, source=DEFAULT_CAMERA_OUTPUT, config=None):
        if source not in ('stabilized', 'execution'):
            raise ValueError('unknown camera output source')
        self.source = source
        self.filter = VirtualStabilizer(config)
        self.latest = None

    @classmethod
    def from_args(cls, args):
        path = getattr(args, 'stabilizer_config', None)
        config = StabilizerConfig(**json.loads(Path(path).read_text())) if path else None
        return cls(getattr(args, 'camera_output', DEFAULT_CAMERA_OUTPUT), config)

    def update(self, t, rp, rr, ap, ar):
        p, r, info = self.filter.update(t, rp, rr, ap, ar)
        return self.accept(t, ap, ar, p, r, info)

    def accept(self, t, ap, ar, sp, sr, info):
        p, r = (sp, sr) if self.source == 'stabilized' else (ap, ar)
        self.latest = dict(t=float(t), camera_output_source=self.source,
            camera_position=np.asarray(p).tolist(),
            camera_quaternion_xyzw=Rotation.from_matrix(r).as_quat().tolist(),
            actual_camera_position=np.asarray(ap).tolist(),
            actual_camera_quaternion_xyzw=Rotation.from_matrix(ar).as_quat().tolist(),
            stabilized_camera_position=np.asarray(sp).tolist(),
            stabilized_camera_quaternion_xyzw=Rotation.from_matrix(sr).as_quat().tolist(),
            stabilization=info)
        return np.asarray(p), np.asarray(r)

    def accept_raw(self, t, position, rotation):
        """Measured camera path: no stabilizer work and no synthetic stabilized pose."""
        if self.source != 'execution':
            raise ValueError('raw camera update requires execution source')
        position = np.asarray(position)
        quaternion = Rotation.from_matrix(rotation).as_quat().tolist()
        self.latest = dict(t=float(t), camera_output_source=self.source,
            camera_position=position.tolist(), camera_quaternion_xyzw=quaternion,
            actual_camera_position=position.tolist(), actual_camera_quaternion_xyzw=quaternion,
            stabilization=dict(enabled=False, reason='raw_physical_camera_input'))
        return position, np.asarray(rotation)

    def pose(self, t=None):
        if self.latest is None or (t is not None and abs(t-self.latest['t']) > 1e-8):
            raise ValueError('no timestamp-matched camera output')
        return np.array(self.latest['camera_position']), Rotation.from_quat(
            self.latest['camera_quaternion_xyzw']).as_matrix()


def video_camera_source(video):
    """Honor a declared output; only pre-contract recordings use the legacy view."""
    manifest = Path(video)/'camera_output.json'
    if manifest.exists():
        source = json.loads(manifest.read_text())['source']
        if source not in ('stabilized', 'execution'):
            raise ValueError('invalid declared camera source')
        if not (Path(video)/source).is_dir():
            raise ValueError(f'declared camera frames are missing: {source}')
        return source
    frames = Path(video)/'frames.json'
    if frames.exists():
        records = json.loads(frames.read_text())
        if records and records[0].get('camera_output_source') == 'stabilized':
            if not (Path(video)/'stabilized').is_dir():
                raise ValueError('stabilized camera frames are missing')
            return 'stabilized'
    return 'execution'


def frame_camera_pose(frame, source=None):
    source = source or frame.get('camera_output_source', 'execution')
    prefix = 'stabilized' if source == 'stabilized' else 'actual'
    return np.asarray(frame[prefix+'_camera_position']), Rotation.from_quat(
        frame[prefix+'_camera_quaternion_xyzw']).as_matrix()


def write_three_camera_video(video, frames, fps):
    """Use each recorded timestamp exactly once; never retime or invent a view."""
    import cv2
    video = Path(video)
    if not frames:
        return None
    first = cv2.imread(str(video/'reference'/f"{frames[0]['frame']:05d}.jpg"))
    if first is None:
        raise ValueError('reference frame missing for three-camera video')
    height, width = first.shape[:2]
    shape = ((height+32+1)//2*2, (3*width+1)//2*2, 3)
    output = video/'comparison.mp4'
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo',
        '-pix_fmt', 'bgr24', '-s', f'{shape[1]}x{shape[0]}', '-r', str(fps), '-i', '-',
        '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18', '-threads', '2',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for frame in frames:
            canvas = np.full(shape, 20, np.uint8)
            for column, view in enumerate(CAMERA_VIEWS):
                image = cv2.imread(str(video/view/f"{frame['frame']:05d}.jpg"))
                if image is None or image.shape != first.shape:
                    raise ValueError(f'missing/mismatched {view} frame {frame["frame"]}')
                x = column*width
                canvas[32:32+height, x:x+width] = image
                label = ('REFERENCE', 'RAW EXECUTION', 'STABILIZED')[column]
                cv2.putText(canvas, f'{label}  t={frame["t"]:.2f}s', (x+6, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, .48, (240, 240, 240), 1, cv2.LINE_AA)
            process.stdin.write(canvas.tobytes())
    finally:
        process.stdin.close()
        error = process.stderr.read().decode(errors='replace')
        process.stderr.close()
        status = process.wait()
    if status:
        raise RuntimeError(f'three-camera video encoder failed: {error}')
    return dict(file=output.name, views=list(CAMERA_VIEWS), frames=len(frames), fps=fps,
                correspondence='same recorded simulation timestamp; original static reference')
