"""Indoor/outdoor classification."""
from vlm_intention.common.env import env_from_scene, env_from_path, parse_label_fields


def test_env_from_scene():
    assert env_from_scene("室内—居家") == "indoor"
    assert env_from_scene("室外自然") == "outdoor"
    assert env_from_scene(None) is None
    assert env_from_scene("住宅") is None     # not a header


def test_env_from_path_fallback():
    assert env_from_path("xx/居家住宿/y.png") == "indoor"
    assert env_from_path("xx/01城市/y.png") == "outdoor"
    assert env_from_path("xx/01城市/y.png", default="indoor") == "indoor"


def test_parse_label_fields():
    s = "场景类别：室内—居家；运镜类型：前推运镜_竖摇；主体：单人"
    f = parse_label_fields(s)
    assert f["场景类别"].startswith("室内")
    assert "前推" in f["运镜类型"]
