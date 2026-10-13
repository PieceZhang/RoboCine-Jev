"""Optional separate ROS process: RGB -> causal SAM 2.1 mask -> small JSON evidence.

Uses the official image predictor on one current image with the previous accepted
box as a prompt. It is a bounded-memory observer, not video lookahead or a general
identity tracker. An operator initializes the subject box; ambiguity is unknown.
Torch and ROS imports stay outside the controller's dependency path.
"""
from concurrent.futures import ThreadPoolExecutor
import time
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np

from .perception import mask_evidence


class SamImageObserver:
    def __init__(self, subject_id, seed_box, checkpoint, config, device='cuda', min_iou=.7):
        box = np.asarray(seed_box, dtype=np.float32)
        if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[:2] < 0) or np.any(box[2:] <= box[:2]):
            raise ValueError('initialize SAM with a valid pixel xyxy subject box')
        self.subject_id, self.box = subject_id, box
        self.checkpoint, self.config, self.device = checkpoint, config, device
        self.min_iou, self.predictor, self.previous_area = min_iou, None, None

    def __call__(self, rgb, stamp):
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        if self.predictor is None:
            self.predictor = SAM2ImagePredictor(build_sam2(self.config, self.checkpoint, device=self.device))
        with torch.inference_mode():
            self.predictor.set_image(rgb)
            masks, scores, _ = self.predictor.predict(box=self.box, multimask_output=True)
        index = int(np.argmax(scores))
        mask, score = masks[index] > 0, float(scores[index])
        evidence = mask_evidence(mask, self.subject_id, stamp, confidence=score if np.isfinite(score) else None)
        area = int(mask.sum())
        # These are segmentation quality checks, never shot success thresholds.
        stable = self.previous_area is None or .4 <= area/max(1, self.previous_area) <= 2.5
        valid = bool(evidence['valid'] and np.isfinite(score) and score >= self.min_iou and stable)
        evidence.update(identity_valid=valid, association='operator_seed_then_previous_mask_box',
                        confidence_kind='SAM_predicted_IoU_not_identity_probability')
        if valid:
            y, x = np.nonzero(mask)
            self.box = np.array([x.min(), y.min(), x.max()+1, y.max()+1], dtype=np.float32)
            self.previous_area = area
        else:
            evidence.update(valid=False, reason='segmentation_or_association_uncertain')
        return evidence


class ExternalSamObserver:
    """Persistent bounded pipe to an existing SAM Python environment (ROS may differ)."""
    def __init__(self, python, subject_id, seed_box, checkpoint, config, device='cuda'):
        env = dict(os.environ)
        package_root = str(Path(__file__).resolve().parents[1])
        # Do not inject ROS Python 3.10 packages into the SAM Python 3.11 process.
        env['PYTHONPATH'] = package_root
        env['PYTHONNOUSERSITE'] = '1'
        config_data = dict(subject_id=subject_id, seed_box=list(seed_box), checkpoint=checkpoint,
                           config=config, device=device)
        self.process = subprocess.Popen([python, '-m', 'jev_agent_wbc.sam_observer',
            '--worker', json.dumps(config_data)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            env=env, bufsize=0)

    def __call__(self, rgb, stamp):
        data = np.ascontiguousarray(rgb, dtype=np.uint8)
        if data.ndim != 3 or data.shape[2] != 3 or data.nbytes > 32*1024*1024:
            raise ValueError('RGB frame exceeds observer input budget')
        self.process.stdin.write(struct.pack('!dIII', stamp, data.shape[0], data.shape[1], data.nbytes))
        self.process.stdin.write(data.tobytes())
        length = struct.unpack('!I', _read_exact(self.process.stdout, 4))[0]
        if length > 65536:
            raise ValueError('invalid observer response size')
        result = json.loads(_read_exact(self.process.stdout, length))
        if 'error' in result:
            raise RuntimeError(result['error'])
        return result

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=1.)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=1.)
        self.process.stdin.close()
        self.process.stdout.close()


def _read_exact(stream, count):
    chunks = bytearray()
    while len(chunks) < count:
        chunk = stream.read(count-len(chunks))
        if not chunk:
            raise EOFError('SAM observer process closed its stream')
        chunks.extend(chunk)
    return bytes(chunks)


def worker_main(config):
    input_stream, output_stream = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr  # model/library diagnostics must not corrupt framing
    observer = SamImageObserver(**config)
    while True:
        try:
            stamp, height, width, length = struct.unpack('!dIII', _read_exact(input_stream, 20))
        except EOFError:
            return
        if length != height*width*3 or not 0 < length <= 32*1024*1024:
            raise ValueError('invalid observer frame')
        rgb = np.frombuffer(_read_exact(input_stream, length), np.uint8).reshape(height, width, 3).copy()
        try:
            result = observer(rgb, stamp)
        except Exception as error:
            result = dict(error=f'{type(error).__name__}: {error}')
        data = json.dumps(result, allow_nan=False).encode()
        output_stream.write(struct.pack('!I', len(data))+data)
        output_stream.flush()


class LatestMaskWorker:
    """One outstanding inference, no unbounded backlog, original capture stamps."""
    def __init__(self, observer):
        self.observer = observer
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='sam-observer')
        self.future = None
        self.dropped = 0

    def submit(self, rgb, stamp):
        if self.future is not None:
            self.dropped += 1
            return False
        self.started = time.monotonic()
        self.future = self.pool.submit(self._observe, rgb.copy(), float(stamp))
        return True

    def _observe(self, rgb, stamp):
        started = time.monotonic()
        result = self.observer(rgb, stamp)
        finished = time.monotonic()
        return result, finished-started, finished

    def poll(self):
        if self.future is None or not self.future.done():
            return None
        future, self.future = self.future, None
        result, inference_s, finished = future.result()
        now = time.monotonic()
        return dict(result, inference_wall_s=inference_s,
                    observation_wall_s=now-self.started, delivery_wait_s=now-finished,
                    dropped_frames=self.dropped)

    def close(self):
        if hasattr(self.observer, 'close'):
            self.observer.close()
        if self.future:
            self.future.cancel()
        self.pool.shutdown(wait=False, cancel_futures=True)


def rgb_from_message(message):
    if message.encoding not in ('rgb8', 'bgr8', 'rgba8', 'bgra8'):
        raise ValueError('SAM observer requires RGB/BGR uint8 camera images')
    channels = 4 if 'a8' in message.encoding else 3
    if message.step < message.width*channels:
        raise ValueError('invalid image stride')
    rows = np.frombuffer(message.data, np.uint8).reshape(message.height, message.step)
    rgb = rows[:, :message.width*channels].reshape(message.height, message.width, channels)[:, :, :3]
    return (rgb[:, :, ::-1] if message.encoding.startswith('bgr') else rgb).copy()


def main(args=None):
    import json
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import ExternalShutdownException
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image
    from std_msgs.msg import String

    class ObserverNode(Node):
        def __init__(self):
            super().__init__('cinematic_sam_observer')
            for key, value in dict(image_topic='/camera/image_raw', subject_id='',
                seed_box=[0., 0., 0., 0.], checkpoint='', config='configs/sam2.1/sam2.1_hiera_b+.yaml',
                device='cuda', sam_python='python3', observation_rate_hz=4.).items():
                self.declare_parameter(key, value)
            value = lambda key: self.get_parameter(key).value
            rate = float(value('observation_rate_hz'))
            if not .1 <= rate <= 10. or not value('subject_id'):
                raise ValueError('subject_id and observation rate [.1, 10] Hz required')
            self.worker = LatestMaskWorker(ExternalSamObserver(value('sam_python'), value('subject_id'), value('seed_box'),
                value('checkpoint'), value('config'), value('device')))
            self.latest = None
            self.last_submitted = -float('inf')
            self.submission_period = 1./rate
            self.last_submission_wall = -float('inf')
            self.publisher = self.create_publisher(String, '/cinematic/subject_visual', 1)
            self.create_subscription(Image, value('image_topic'), self.on_image, qos_profile_sensor_data)
            # Deliver finished evidence promptly without increasing GPU rate.
            self.create_timer(min(.05, self.submission_period), self.tick)

        def on_image(self, message):
            self.latest = message

        def tick(self):
            try:
                result = self.worker.poll()
                if result is not None:
                    msg = String()
                    msg.data = json.dumps(result, allow_nan=False)
                    self.publisher.publish(msg)
                message = self.latest
                now = time.monotonic()
                if (message is not None and self.worker.future is None and
                        now-self.last_submission_wall >= self.submission_period):
                    stamp = message.header.stamp.sec+message.header.stamp.nanosec*1e-9
                    if stamp > self.last_submitted:
                        self.worker.submit(rgb_from_message(message), stamp)
                        self.last_submitted = stamp
                        self.last_submission_wall = now
            except Exception as error:
                self.get_logger().error(str(error))

    rclpy.init(args=args)
    node = ObserverNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.worker.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        worker_main(json.loads(sys.argv[2]))
    else:
        main()
