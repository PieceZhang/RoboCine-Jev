"""Portable episodes in the existing CPU and PhysX fixture format."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from jev_agent_wbc.contracts import validate_task
from jev_agent_wbc.reference import OriginalReference
from jev_agent_wbc.scenarios import RuntimeScene


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def fingerprint(task, scene):
    return hashlib.sha256(json.dumps([task, scene], sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def source_fingerprints():
    package = Path(__file__).resolve().parents[1]
    paths = list((package/'CineGym').glob('*.py'))
    # Acceptance and continuous witnesses depend on more than the nominal
    # controller. Record the complete Python package, including those gates.
    paths += list((package/'jev_agent_wbc').glob('*.py'))
    return {str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


@dataclass
class Episode:
    task: dict
    scene: dict
    metadata: dict

    @property
    def reference(self):
        return OriginalReference(self.task, self.scene['subject']['position'])

    @property
    def runtime(self):
        return RuntimeScene(self.scene)

    def save(self, folder):
        validate_task(self.task)
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=False)
        write_json(folder/'task.json', self.task)
        write_json(folder/'runtime_scene.json', self.scene)
        write_json(folder/'metadata.json', {**self.metadata, 'sha256': fingerprint(self.task, self.scene)})
        return folder

    @classmethod
    def load(cls, folder):
        folder = Path(folder)
        task, scene, metadata = [json.loads((folder/name).read_text())
                                 for name in ('task.json', 'runtime_scene.json', 'metadata.json')]
        validate_task(task)
        if metadata.get('sha256') != fingerprint(task, scene):
            raise ValueError('episode fingerprint mismatch: task or scene was edited')
        return cls(task, scene, metadata)
