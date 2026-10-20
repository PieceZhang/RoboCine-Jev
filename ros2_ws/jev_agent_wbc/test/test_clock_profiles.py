import numpy as np
from jev_agent_wbc.clock_profiles import profiles


def test_clock_profiles_include_wait_then_recovery_and_respect_rate_acceleration():
    schedules=profiles(1.,2.,4.,12,.1)
    for p in schedules:
        assert np.all(p['increments']>=0.)
        assert np.all((p['rates']>=0.)&(p['rates']<=2.5))
        assert np.max(abs(np.diff(np.r_[1.,p['rates']])))<=4/12+1e-10
        assert abs(p['next_rate']-1.)<=.1+1e-10
    waiting=next(p for p in schedules if p['name']=='yield_long')
    assert min(waiting['rates'])==0. and waiting['rates'][-1]>2.
    assert waiting['next_rate']<1.


def test_first_command_does_not_apply_a_later_profile_switch_early():
    p=next(p for p in profiles(.2,2.5,4.,12,.02) if p['name']=='yield_short')
    assert np.isclose(p['next_rate'],.18)
    assert p['rates'][-1]>2.
