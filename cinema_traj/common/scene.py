"""Scene graph: rooms / structures / objects / doors / optional subject track.

The scene JSON is the CANONICAL input of the whole pipeline. It is either
hand-authored (demo scenes) or drafted by the extract stage from a PCD/PLY
map and then hand-refined. Schema (all rotations are 3x3 row-major nested
arrays with local axes as COLUMNS):

    {
      "scene_id": "...",
      "rooms":       [{id, label, bounds_min[3], bounds_max[3]}],
      "structures":  [{id, kind: wall|floor|ceiling, room, obb}],
      "objects":     [{id, label, room, obb,
                       placement: freestanding|wall|ceiling|floor_wall,
                       attached_structure?: id}],
      "doors":       [{id, kind: door|doorway|archway, rooms[2], obb}],
      "subject_track"?: {...}          # see cinema_traj.common.subject
    }
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .obb import OBB
from .subject import SubjectTrack

PLACEMENTS = ("freestanding", "wall", "ceiling", "floor_wall")
STRUCTURE_KINDS = ("wall", "floor", "ceiling")
DOOR_KINDS = ("door", "doorway", "archway")


@dataclass
class Room:
    id: str
    label: str
    bounds_min: np.ndarray
    bounds_max: np.ndarray

    @property
    def center(self) -> np.ndarray:
        return (self.bounds_min + self.bounds_max) / 2.0

    @property
    def size(self) -> np.ndarray:
        return self.bounds_max - self.bounds_min

    def contains(self, p) -> bool:
        q = np.asarray(p, dtype=np.float64)
        return bool(np.all(q >= self.bounds_min) and np.all(q <= self.bounds_max))

    @classmethod
    def from_dict(cls, d: dict) -> "Room":
        return cls(d["id"], d.get("label", d["id"]),
                   np.asarray(d["bounds_min"], float), np.asarray(d["bounds_max"], float))


@dataclass
class StructureNode:
    id: str
    kind: str                  # wall | floor | ceiling
    room: str
    obb: OBB

    @classmethod
    def from_dict(cls, d: dict) -> "StructureNode":
        return cls(d["id"], d["kind"], d.get("room", ""), OBB.from_dict(d["obb"]))


@dataclass
class ObjectNode:
    id: str
    label: str
    room: str
    obb: OBB
    placement: str = "freestanding"          # freestanding | wall | ceiling | floor_wall
    attached_structure: str | None = None
    viewing_distance: float | None = None    # optional manual override

    @classmethod
    def from_dict(cls, d: dict) -> "ObjectNode":
        return cls(d["id"], d.get("label", d["id"]), d.get("room", ""), OBB.from_dict(d["obb"]),
                   d.get("placement", "freestanding"), d.get("attached_structure"),
                   d.get("viewing_distance"))

    def to_dict(self) -> dict:
        d = {"id": self.id, "label": self.label, "room": self.room,
             "obb": self.obb.to_dict(), "placement": self.placement}
        if self.attached_structure:
            d["attached_structure"] = self.attached_structure
        if self.viewing_distance is not None:
            d["viewing_distance"] = self.viewing_distance
        return d


@dataclass
class DoorNode:
    id: str
    kind: str                  # door | doorway | archway
    rooms: list                # [room_a, room_b]
    obb: OBB
    traversable: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "DoorNode":
        return cls(d["id"], d.get("kind", "door"), list(d.get("rooms", [])),
                   OBB.from_dict(d["obb"]), d.get("traversable", True))


@dataclass
class Scene:
    scene_id: str
    rooms: list = field(default_factory=list)
    structures: list = field(default_factory=list)
    objects: list = field(default_factory=list)
    doors: list = field(default_factory=list)
    subject: SubjectTrack | None = None
    provenance: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Index helpers
    # ------------------------------------------------------------------ #
    def room_of(self, obj: ObjectNode) -> Room | None:
        for r in self.rooms:
            if r.id == obj.room:
                return r
        return None

    def object_by_id(self, oid: str) -> ObjectNode | None:
        for o in self.objects:
            if o.id == oid:
                return o
        return None

    def collision_obbs(self) -> list[OBB]:
        """All OBBs the camera must not pass through."""
        return [s.obb for s in self.structures] + [o.obb for o in self.objects]

    def room_bounds(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(r.bounds_min, r.bounds_max) for r in self.rooms]

    # ------------------------------------------------------------------ #
    # IO
    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, d: dict) -> "Scene":
        subj = SubjectTrack.from_dict(d["subject_track"]) if d.get("subject_track") else None
        return cls(
            scene_id=d.get("scene_id", "scene"),
            rooms=[Room.from_dict(r) for r in d.get("rooms", [])],
            structures=[StructureNode.from_dict(s) for s in d.get("structures", [])],
            objects=[ObjectNode.from_dict(o) for o in d.get("objects", [])],
            doors=[DoorNode.from_dict(x) for x in d.get("doors", [])],
            subject=subj,
            provenance=d.get("provenance", {}),
        )

    def to_dict(self) -> dict:
        d = {
            "scene_id": self.scene_id,
            "rooms": [{"id": r.id, "label": r.label,
                       "bounds_min": r.bounds_min.tolist(),
                       "bounds_max": r.bounds_max.tolist()} for r in self.rooms],
            "structures": [{"id": s.id, "kind": s.kind, "room": s.room,
                            "obb": s.obb.to_dict()} for s in self.structures],
            "objects": [o.to_dict() for o in self.objects],
            "doors": [{"id": x.id, "kind": x.kind, "rooms": x.rooms,
                       "obb": x.obb.to_dict(), "traversable": x.traversable}
                      for x in self.doors],
        }
        if self.subject is not None:
            d["subject_track"] = self.subject.to_dict()
        if self.provenance:
            d["provenance"] = self.provenance
        return d

    # ------------------------------------------------------------------ #
    # Validation & LLM summary
    # ------------------------------------------------------------------ #
    def validate(self) -> list[str]:
        errs: list[str] = []
        room_ids = {r.id for r in self.rooms}
        ids = [r.id for r in self.rooms] + [s.id for s in self.structures] + \
              [o.id for o in self.objects] + [x.id for x in self.doors]
        if len(ids) != len(set(ids)):
            errs.append("duplicate ids in scene graph")
        for r in self.rooms:
            if np.any(r.bounds_max <= r.bounds_min):
                errs.append(f"room {r.id}: bounds_max must exceed bounds_min")
        for s in self.structures:
            if s.kind not in STRUCTURE_KINDS:
                errs.append(f"structure {s.id}: kind must be one of {STRUCTURE_KINDS}")
            if s.room and s.room not in room_ids:
                errs.append(f"structure {s.id}: unknown room '{s.room}'")
        for o in self.objects:
            if o.room and o.room not in room_ids:
                errs.append(f"object {o.id}: unknown room '{o.room}'")
            if o.placement not in PLACEMENTS:
                errs.append(f"object {o.id}: placement must be one of {PLACEMENTS}")
            if o.attached_structure and not any(
                    s.id == o.attached_structure for s in self.structures):
                errs.append(f"object {o.id}: attached_structure "
                            f"'{o.attached_structure}' not found")
        for x in self.doors:
            if x.kind not in DOOR_KINDS:
                errs.append(f"door {x.id}: kind must be one of {DOOR_KINDS}")
            for rid in x.rooms:
                if rid not in room_ids:
                    errs.append(f"door {x.id}: unknown room '{rid}'")
        if self.subject is not None:
            ts = [f["t"] for f in self.subject.frames]
            if any(b <= a for a, b in zip(ts, ts[1:])):
                errs.append("subject_track: frame times must be strictly increasing")
        return errs

    def summary(self) -> str:
        """Compact textual scene summary for the LLM planner prompt."""
        lines = []
        for r in self.rooms:
            c = r.center
            lines.append(f"room {r.id} ({r.label}): center=({c[0]:.1f},{c[1]:.1f},{c[2]:.1f}) "
                         f"size=({r.size[0]:.1f}x{r.size[1]:.1f}x{r.size[2]:.1f}m)")
        for o in self.objects:
            c = o.obb.center
            e = o.obb.extents
            lines.append(f"object {o.id} '{o.label}' room={o.room} "
                         f"center=({c[0]:.1f},{c[1]:.1f},{c[2]:.1f}) "
                         f"size=({e[0]:.1f}x{e[1]:.1f}x{e[2]:.1f}m) placement={o.placement}")
        for x in self.doors:
            c = x.obb.center
            lines.append(f"door {x.id} kind={x.kind} rooms={x.rooms} "
                         f"center=({c[0]:.1f},{c[1]:.1f},{c[2]:.1f})")
        if self.subject is not None:
            s = self.subject.summary()
            lines.append(f"subject {s['subject_id']}: height={s['height_m']}m "
                         f"moves {s['start_pos']}→{s['end_pos']} over {s['duration_s']}s "
                         f"(heading {s['start_heading_deg']}°→{s['end_heading_deg']}°)")
        return "\n".join(lines)


def load_scene(path) -> Scene:
    """Load + validate a scene JSON; raises on schema errors."""
    import json
    from pathlib import Path

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    scene = Scene.from_dict(data)
    errs = scene.validate()
    if errs:
        raise ValueError("invalid scene JSON:\n  - " + "\n  - ".join(errs))
    return scene
