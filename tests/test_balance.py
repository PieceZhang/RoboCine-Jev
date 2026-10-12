"""Balance stage: greedy selection reaches >= target rate on every direction."""
from vlm_intention.common.schema import DIRS
from vlm_intention.stages import balance


def _synthetic(n_pos, n_neg=200):
    """n_pos frames each carrying ALL 5 directions True; n_neg all-negative."""
    items = {}
    for i in range(n_pos):
        items[f"pos/{i}.png"] = {d: True for d in DIRS}
    for i in range(n_neg):
        items[f"neg/{i}.png"] = {d: False for d in DIRS}
    return items


def test_balance_reaches_target():
    merged = _synthetic(n_pos=50, n_neg=200)
    selected, target, need, cur = balance.build(merged, target_rate=0.35)
    n = len(selected)
    assert n == target
    for d in DIRS:
        assert cur[d] / n >= 0.35            # every direction meets the target
    assert cur["前"] == 50                    # all positives are multi-direction here


def test_target_bounded_by_rarest():
    # rarest direction = 40 positives -> target = floor(40/0.35) = 114
    merged = {f"k{i}.png": {d: (i < 40) for d in DIRS} for i in range(200)}
    selected, target, need, cur = balance.build(merged, target_rate=0.35)
    assert target == 114
