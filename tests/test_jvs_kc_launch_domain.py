"""Both halves of KC-WBC must isolate ROS traffic before starting children."""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("launch")
pytest.importorskip("launch_ros")
from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, OpaqueFunction, SetEnvironmentVariable


LAUNCH_DIR = Path(__file__).resolve().parents[1] / "ros2_ws/jvs_mock/launch"


@pytest.mark.parametrize("filename", [
    "jvs_kc_wbc_isaac.launch.py", "jvs_kc_wbc_trajfollow.launch.py",
])
@pytest.mark.parametrize("override,expected", [(None, "176"), ("201", "201"), ("0", "0")])
def test_kc_domain_is_set_before_any_child_even_with_ambient_domain_zero(filename, override, expected):
    spec = importlib.util.spec_from_file_location("kc_launch", LAUNCH_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    context = LaunchContext()
    context.environment["ROS_DOMAIN_ID"] = "0"
    description = module.generate_launch_description()
    for entity in description.entities:
        if isinstance(entity, DeclareLaunchArgument):
            entity.execute(context)
    # Only existence is needed during launch expansion; don't depend on a dataset.
    context.launch_configurations["stage_view"] = str(Path(__file__).resolve())
    if override is not None:
        context.launch_configurations["ros_domain_id"] = override
    setup = next(e for e in description.entities if isinstance(e, OpaqueFunction))
    actions = setup.execute(context)
    assert isinstance(actions[0], SetEnvironmentVariable)
    actions[0].execute(context)
    assert context.environment["ROS_DOMAIN_ID"] == expected


@pytest.mark.parametrize("filename", [
    "jvs_kc_wbc_isaac.launch.py", "jvs_kc_wbc_trajfollow.launch.py",
])
@pytest.mark.parametrize("domain", ["-1", "233", "not-a-number"])
def test_kc_rejects_invalid_domain_before_starting_children(filename, domain):
    spec = importlib.util.spec_from_file_location("kc_launch", LAUNCH_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    context = LaunchContext()
    context.launch_configurations["ros_domain_id"] = domain
    setup = (module._start_isaac if "isaac.launch" in filename else module._runtime_actions)
    with pytest.raises(ValueError):
        setup(context, str(LAUNCH_DIR.parent))
