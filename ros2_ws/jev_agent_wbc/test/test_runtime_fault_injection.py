import pytest
from jev_agent_wbc.runtime_fault_injection import FaultSchedule


def test_faults_are_explicit_once_and_gaps_have_fixed_physical_boundaries():
    events=[dict(kind='controller_delay',time_s=2.,duration_s=.15),
            dict(kind='controller_error',time_s=3.),dict(kind='lidar_gap',time_s=4.,duration_s=.3)]
    schedule=FaultSchedule(events)
    assert not schedule.due('controller_delay',1.99)
    assert schedule.due('controller_delay',2.08)==events[:1]
    assert not schedule.due('controller_delay',3.)
    assert schedule.due('controller_error',3.08)==events[1:2]
    assert schedule.active('lidar_gap',4.) and schedule.active('lidar_gap',4.299)
    assert not schedule.active('lidar_gap',4.3)


@pytest.mark.parametrize('event',[dict(kind='rewind',time_s=0.),
    dict(kind='controller_delay',time_s=0.,duration_s=-1.),
    dict(kind='lidar_gap',time_s=float('nan'),duration_s=.1)])
def test_invalid_fault_script_is_rejected_before_running_physics(event):
    with pytest.raises(ValueError):FaultSchedule([event])
