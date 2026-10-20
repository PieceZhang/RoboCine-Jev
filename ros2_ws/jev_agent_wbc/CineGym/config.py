"""Reproducible sampling domains; difficulty never changes acceptance gates."""
from dataclasses import dataclass
import math

TEMPLATES = {
    'orbit': ('orbit',), 'movein': ('movein',),
    'truck_left': ('truck_left',), 'truck_right': ('truck_right',),
    'pedestal_up': ('pedestal_up',), 'pedestal_down': ('pedestal_down',),
    'combo_01_easy': ('movein', 'truck_left'),
    'combo_02_easy': ('truck_right', 'pedestal_up'),
    'combo_03_medium': ('movein', 'orbit', 'pedestal_up'),
    'combo_04_medium': ('truck_left', 'movein', 'pedestal_down'),
    'combo_05_hard': ('orbit', 'movein', 'truck_right'),
    'combo_06_hard': ('pedestal_up', 'truck_left', 'pedestal_down'),
}
FAMILIES = tuple(TEMPLATES) + ('moveout', 'random_combo')
MOTIONS = ('crossing', 'curved', 'stop_go', 'reverse', 'wave', 'diagonal3d')
LEVELS = ('low', 'middle', 'high')
DIFFICULTIES = ('easy', 'medium', 'hard')
PROFILES = {
    'easy': dict(count=(3, 3), penetration=(.015, .035), speed=(.16, .30)),
    'medium': dict(count=(3, 5), penetration=(.035, .065), speed=(.22, .42)),
    'hard': dict(count=(5, 7), penetration=(.055, .09), speed=(.30, .58)),
}


@dataclass(frozen=True)
class GymConfig:
    difficulty: str = 'medium'
    family: str = 'mixed'
    # Natural scenes mix independent passing actors and actual conflicts.
    # Legacy preserves the original all-actors-intersect distribution.
    generation_style: str = 'natural'
    require_avoidance_witness: bool = False
    # Ground boxes always require a complete mixed-scene timing witness.
    ground_boxes: bool = True
    obstacle_count: int = None
    # Limit random_combo without filtering episodes after seeing their outcome.
    combo_actions: int = None
    # Optional authoring calibration; does not change execution acceptance.
    enforce_global_penetration: bool = False
    # Per main action; combos have 2--4 actions with shared endpoints.
    action_duration_s: tuple = (8., 12.)
    radius_m: tuple = (.075, .125)
    sample_step_s: float = .1
    audit_step_s: float = .02
    initial_clearance_m: float = .12
    max_obstacle_speed_m_s: float = 1.2
    max_obstacle_acceleration_m_s2: float = 3.
    max_attempts: int = 24
    split: str = 'train'

    def __post_init__(self):
        if self.generation_style not in ('natural', 'legacy'):
            raise ValueError('generation_style must be natural or legacy')
        if self.difficulty not in DIFFICULTIES or self.family not in (*FAMILIES, 'mixed'):
            raise ValueError('unknown difficulty or camera family')
        if self.split not in ('train', 'validation', 'test'):
            raise ValueError('split must be train, validation or test')
        if self.obstacle_count is not None and (type(self.obstacle_count) is not int or not 3 <= self.obstacle_count <= 12):
            raise ValueError('obstacle_count must be 3..12 (all three height levels)')
        if self.combo_actions is not None and (type(self.combo_actions) is not int or not 2 <= self.combo_actions <= 4):
            raise ValueError('combo_actions must be 2..4')
        if type(self.enforce_global_penetration) is not bool:
            raise ValueError('enforce_global_penetration must be bool')
        if type(self.require_avoidance_witness) is not bool:
            raise ValueError('require_avoidance_witness must be bool')
        if type(self.ground_boxes) is not bool:
            raise ValueError('ground_boxes must be bool')
        for name, bounds in (('action_duration_s', (4., 20.)), ('radius_m', (.04, .18))):
            value = getattr(self, name)
            if len(value) != 2 or not all(math.isfinite(x) for x in value) or not bounds[0] <= value[0] <= value[1] <= bounds[1]:
                raise ValueError(f'invalid {name}: expected ordered range within {bounds}')
        for name in ('sample_step_s', 'audit_step_s', 'initial_clearance_m',
                     'max_obstacle_speed_m_s', 'max_obstacle_acceleration_m_s2'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be positive and finite')
        if self.sample_step_s > .2 or self.audit_step_s > .05:
            raise ValueError('reference/audit intervals must be <= .2/.05 seconds')
        if type(self.max_attempts) is not int or self.max_attempts < 1:
            raise ValueError('max_attempts must be a positive integer')
