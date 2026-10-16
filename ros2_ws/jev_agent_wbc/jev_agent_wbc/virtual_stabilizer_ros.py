"""Independent DDS camera backend. Ordered physics batches keep exact sim time.

Only publishes camera poses; never publishes actuator commands. JSON batches
carry the two poses together to avoid pairing different reference timestamps.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from .virtual_stabilizer import StabilizerConfig, VirtualStabilizer


class StabilizerNode(Node):
    def __init__(self, config, journal):
        super().__init__('virtual_camera_stabilizer')
        self.config = config
        self.filter = VirtualStabilizer(config)
        self.publisher = self.create_publisher(String, '/virtual_camera/output', 10)
        self.create_subscription(String, '/virtual_camera/input', self.receive, 10)
        self.journal = Path(journal).open('w') if journal else None
        self.session = None

    def receive(self, message):
        request = json.loads(message.data)
        if request['session'] != self.session:
            self.filter.reset()
            self.session = request['session']
        result = dict(session=self.session, sequence=request['sequence'], samples=[])
        try:
            for row in request['samples']:
                rp, rr, ap, ar = (row[k] for k in ('rp', 'rr', 'ap', 'ar'))
                p, r, info = self.filter.update(row['t'], rp, rr, ap, ar)
                result['samples'].append(dict(t=row['t'], p=p.tolist(), r=r.tolist(), info=info))
            if self.journal:
                self.journal.write(json.dumps(dict(request=request, response=result))+'\n')
                self.journal.flush()
        except Exception as error:
            result['error'] = f'{type(error).__name__}: {error}'
        self.publisher.publish(String(data=json.dumps(result)))

    def destroy_node(self):
        if self.journal:
            self.journal.close()
        super().destroy_node()


class StabilizerClient:
    def __init__(self, session, timeout=30.):
        self.owns_context = not rclpy.ok()
        if self.owns_context:
            rclpy.init(args=None)
        self.node = rclpy.create_node('virtual_camera_renderer')
        self.publisher = self.node.create_publisher(String, '/virtual_camera/input', 10)
        self.node.create_subscription(String, '/virtual_camera/output', self.receive, 10)
        self.session, self.sequence, self.response = session, 0, None
        deadline = time.monotonic()+timeout
        while self.publisher.get_subscription_count() == 0:
            if time.monotonic() >= deadline:
                raise TimeoutError('no independent ROS virtual-camera stabilizer discovered')
            rclpy.spin_once(self.node, timeout_sec=.05)

    def receive(self, message):
        response = json.loads(message.data)
        if response['session'] == self.session and response['sequence'] == self.sequence:
            self.response = response

    def update(self, samples):
        self.sequence += 1
        self.response = None
        request = dict(session=self.session, sequence=self.sequence, samples=samples)
        self.publisher.publish(String(data=json.dumps(request)))
        deadline = time.monotonic()+10.
        while self.response is None:
            if time.monotonic() >= deadline:
                raise TimeoutError('missing timestamp-matched stabilized camera response')
            rclpy.spin_once(self.node, timeout_sec=.02)
        if 'error' in self.response:
            raise RuntimeError(self.response['error'])
        return self.response['samples']

    def close(self):
        self.node.destroy_node()
        if self.owns_context:
            rclpy.shutdown()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path)
    p.add_argument('--journal', type=Path)
    args = p.parse_args()
    config = StabilizerConfig(**json.loads(args.config.read_text())) if args.config else StabilizerConfig()
    rclpy.init(args=[])
    node = StabilizerNode(config, args.journal)
    print(json.dumps(dict(event='VIRTUAL_STABILIZER_READY', domain=os.getenv('ROS_DOMAIN_ID'), config=asdict(config))), flush=True)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
