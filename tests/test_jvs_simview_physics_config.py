"""Scene runners accept the same solver and finite actuator settings."""
import importlib.util
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'ros2_ws/jvs_mock/scripts'
sys.path.insert(0, str(SCRIPTS))


def runner(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('name', ['jvs_simview_isaac', 'jvs_dynamic_simview_isaac'])
def test_scene_runner_can_select_pgs_or_tgs_with_explicit_servo_settings(name):
    module = runner(name)
    parser = module.build_parser()
    assert parser.get_default('solver_type') == 'PGS'
    for solver in ['PGS', 'TGS']:
        args = parser.parse_args(['--xacro', str(SCRIPTS / 'jvs_isaac_sim.py'),
            '--simview-stage', str(SCRIPTS / 'jvs_isaac_sim.py'),
            '--xacro-executable', str(SCRIPTS / 'jvs_isaac_sim.py'),
            '--solver-type', solver, '--solver-velocity-iterations', '8',
            '--arm-drive-stiffness', '300', '--arm-drive-damping', '60',
            '--wheel-drive-damping', '10', '--actuator-armature', '.001'])
        module.validate_args(args)
        assert args.solver_type == solver
        assert (args.solver_velocity_iterations, args.arm_drive_stiffness,
                args.arm_drive_damping, args.wheel_drive_damping, args.actuator_armature) == (8, 300., 60., 10., .001)


@pytest.mark.parametrize('option,value', [('--arm-drive-stiffness', 'nan'),
    ('--arm-drive-damping', '-1'), ('--wheel-drive-damping', 'inf'),
    ('--actuator-armature', '-.1'), ('--solver-velocity-iterations', '0'),
    ('--physics-hz', 'nan'), ('--command-filter-hz','-1'), ('--command-filter-hz','inf')])
def test_dynamic_runner_rejects_invalid_physics_before_starting_kit(option, value):
    module = runner('jvs_dynamic_simview_isaac')
    args = module.build_parser().parse_args(['--xacro', str(SCRIPTS / 'jvs_isaac_sim.py'),
        '--simview-stage', str(SCRIPTS / 'jvs_isaac_sim.py'), option, value])
    with pytest.raises(ValueError):
        module.validate_args(args)


def test_angular_drive_units_produce_the_same_torque_for_radians_and_degrees():
    import math
    module = runner('jvs_isaac_sim')
    # A 0.1 rad error at 1000 Nm/rad must generate 100 Nm, not 5729.6 Nm.
    assert module.angular_drive_gain_to_usd(1000.)*math.degrees(.1) == pytest.approx(100.)
    # The same conversion applies to the wheel's damping and angular speed.
    assert module.angular_drive_gain_to_usd(20.)*math.degrees(.2) == pytest.approx(4.)
    assert module.angular_drive_gain_to_usd(0.) == 0.


@pytest.mark.parametrize('name', ['jvs_isaac_sim', 'jvs_simview_isaac', 'jvs_dynamic_simview_isaac'])
def test_runners_default_to_the_validated_coordinated_physics_profile(name):
    parser = runner(name).build_parser()
    for key, value in dict(solver_type='PGS', solver_velocity_iterations=16,
            arm_drive_stiffness=200., arm_drive_damping=1000., wheel_drive_damping=20.,
            command_filter_hz=0., physics_feedback_hz=50., arm_drive_type='force').items():
        assert parser.get_default(key) == value


@pytest.mark.parametrize('filename', ['jvs_mock_isaac.launch.py', 'jvs_mock_simview.launch.py',
    'jvs_mock_dynamic_simview.launch.py', 'jvs_kc_wbc_isaac.launch.py',
    'jvs_kc_wbc_dynamic_isaac.launch.py'])
def test_launch_defaults_match_the_same_physics_profile(filename):
    from launch import LaunchContext
    from launch.actions import DeclareLaunchArgument
    spec = importlib.util.spec_from_file_location('stability_launch', SCRIPTS.parent/'launch'/filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    context = LaunchContext()
    context.launch_configurations.update(stage_view=str(SCRIPTS), simview_stage=str(SCRIPTS))
    for entity in module.generate_launch_description().entities:
        if isinstance(entity, DeclareLaunchArgument):
            entity.execute(context)
    assert context.launch_configurations['solver_type'] == 'PGS'
    assert context.launch_configurations['arm_drive_type'] == 'force'
    for key, value in dict(solver_velocity_iterations=16, arm_drive_stiffness=200.,
            arm_drive_damping=1000., wheel_drive_damping=20., command_filter_hz=0.,
            physics_feedback_hz=50.).items():
        assert float(context.launch_configurations[key]) == value
