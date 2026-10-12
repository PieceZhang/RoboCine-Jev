#!/usr/bin/env python3
r"""Rewrite annotation-JSON keys + nested relative_path after a layout flatten.

Maintenance util (NOT a pipeline stage). Pattern/replacement/files are
config-driven (reorg section). Idempotent: skips already-migrated keys.

Default (current dataset): 分场景切片\d+-\d+/  →  横屏/
  python scripts/fix_paths_after_reorg.py [--config config.yaml]
"""
import json
import os
import re
import shutil
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from vlm_intention.common.config import load_config
from vlm_intention.common.paths import resolve_base

_backed = set()


def fix_str(s, pat):
    return pat.sub(_REPL, s) if isinstance(s, str) else s


def process(base, name, pat):
    p = os.path.join(str(base), name)
    if not os.path.exists(p):
        print(f"  ⚠ 跳过(不存在): {name}"); return 0
    data = json.load(open(p, encoding="utf-8"))
    bak = p + ".bak-prereorg"
    if not os.path.exists(bak):
        shutil.copy2(p, bak)
    out, changed = {}, 0
    for k, v in data.items():
        nk = fix_str(k, pat)
        if isinstance(v, dict) and "relative_path" in v:
            v = dict(v); v["relative_path"] = fix_str(v["relative_path"], pat)
        out[nk] = v
        changed += nk != k
    tmp = p + ".tmp"
    json.dump(out, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    os.replace(tmp, p)
    miss = [k for k in out if not k.startswith("_")
            and not os.path.exists(os.path.join(str(base), k))]
    print(f"  {'✅' if not miss else '❌'} {name}: {len(out)} 条 | 改 key {changed} | 缺失 {len(miss)}")
    if miss[:5]:
        print("      缺失示例:", miss[:5])
    return len(miss)


_REPL = None   # set in main (config-driven)


def main():
    global _REPL
    cfg = load_config()
    base = resolve_base(cfg)
    _REPL = cfg.reorg.replacement
    pat = re.compile(cfg.reorg.pattern)
    print(f"=== 修复路径: {cfg.reorg.pattern} → {cfg.reorg.replacement} ===")
    total = sum(process(base, f, pat) for f in cfg.reorg.files)
    print(f"\n{'✅ 全部对上' if total == 0 else '⚠️ 仍有缺失'} | 总缺失 {total}")


if __name__ == "__main__":
    main()
