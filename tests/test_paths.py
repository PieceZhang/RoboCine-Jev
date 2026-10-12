"""Pin mp4_for / mp4_rel_for / safe_resolve against known keys (migration safety)."""
from vlm_intention.common.paths import mp4_for, mp4_rel_for, safe_resolve


def test_mp4_for_keyframe():
    assert mp4_for("横屏/01城市/标注tag/jimeng-x.png") == "横屏/01城市/jimeng-x.mp4"


def test_mp4_rel_for_mined():
    # 抽帧_full extracted frame -> source mp4 (the WORKING algorithm; the
    # original serve_mined_gt copy had a buggy double-reassign here).
    assert mp4_rel_for("抽帧_full/横屏/11花海/jimeng-x_f02.png") == "横屏/11花海/jimeng-x.mp4"
    assert mp4_rel_for("抽帧_pilot/02城市夜景/jimeng-y_f00.png") == "02城市夜景/jimeng-y.mp4"


def test_safe_resolve_blocks_traversal(tmp_path):
    (tmp_path / "ok.png").write_bytes(b"x")
    full, ok = safe_resolve(tmp_path, "ok.png")
    assert ok and full.endswith("ok.png")
    full, ok = safe_resolve(tmp_path, "../etc/passwd")
    assert not ok
    full, ok = safe_resolve(tmp_path, "missing.png")
    assert not ok
