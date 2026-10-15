"""Bounded search choices. None of these settings changes an execution limit."""
import numpy as np
from scipy.optimize import linprog


PROFILES = {
    'executable': dict(description='Generate camera targets from collision-constrained robot motion, then strictly reverify the explicit FK-derived path. Report all shot deviations.',
                       amplitudes=(.12,.18,.20,.30),allocations=('path_heading','anticipatory','base_assisted','arm_assisted')),
    'local': dict(description='Small nearby deformations; preserve measured bent posture; explore both timing signs.',
                  amplitudes=(.12, .18, .20, .22), allocations=('anticipatory','path_heading','arm_assisted','base_assisted')),
    'base': dict(description='Use base-assisted continuous FK allocations first; explore moderate returning detours.',
                 amplitudes=(.12, .18, .22, .30), allocations=('anticipatory','base_assisted','path_heading','arm_assisted')),
    'timing': dict(description='Prefer time redistribution with unchanged deadline and small spatial repairs.',
                   amplitudes=(.08, .12, .18, .22), allocations=('anticipatory','path_heading','balanced','base_assisted')),
    'explore': dict(description='Broaden the nearby spatial search after failed local probes; retain all physical gates.',
                    amplitudes=(.18, .20, .30, .40), allocations=('anticipatory','path_heading','base_assisted','arm_assisted')),
}


def profile(name):
    if name is None:
        return None
    if name not in PROFILES:
        raise ValueError('unknown bounded search configuration')
    return PROFILES[name]


def capability(search, subject):
    """Directional coupled actuator LPs; a local prior, never a path certificate."""
    prior = search.prior()
    radial = search.p-np.asarray(subject); radial[2] = 0.
    radial /= max(np.linalg.norm(radial), 1e-9)
    left = np.cross(radial, [0.,0.,1.])
    directions = dict(outer=radial, inner=-radial, left=left, right=-left,
                      above=np.array([0.,0.,1.]), below=np.array([0.,0.,-1.]))
    a, lo, hi = (np.asarray(prior[k]) for k in ('control_matrix','control_lower','control_upper'))
    probes = []
    for name, direction in directions.items():
        result = None if prior['error'] else linprog(-direction@search.j[:3],
            A_ub=np.vstack([a,-a]), b_ub=np.r_[hi,-lo], bounds=[(None,None)]*8, method='highs')
        probes.append(dict(direction=name, translation_support_m_s=(
            max(0.,float(-result.fun)) if result is not None and result.success else None)))
    return dict(snapshot_time_s=search.t, horizon_s=search.policy.prior_horizon_s,
                probes=probes, initial_error=prior['error'],
                interpretation='coupled local velocity support, orientation unconstrained; not reachability or safety',
                velocity_source=search.velocity_source)


def offers(family, capability_summary, feedback=()):
    return dict(selected_family=family, robot_capability=capability_summary,
        latest_search_feedback=list(feedback)[-3:],
        configurations={k:dict(description=v['description'], amplitude_seeds_m=list(v['amplitudes']),
                              allocation_order=list(v['allocations'])) for k,v in PROFILES.items()},
        immutable='collision margins, FK tolerance, actuator limits, original task and deadline')
