"""标注结果汇总：维度分布、培体统计、失败/夹紧清单 -> report.json + summary.csv。"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


def load_records(annotations_path: Path) -> list[dict]:
    records = []
    if annotations_path.exists():
        for line in annotations_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed annotation JSONL: {annotations_path}") from exc
    return records


def _split_last(values: list[str | None], split: bool) -> Counter:
    """值按 / 取小类部分、按 _ 拆多选后计数；None 计入 <缺失>。"""
    c: Counter = Counter()
    for v in values:
        if v is None:
            c["<缺失>"] += 1
            continue
        keys = v.split("/", 1)[1].split("_") if split and "/" in v else [v]
        for k in keys:
            if k:  # 培体空串不计
                c[k] += 1
    if split:
        c.setdefault("<空>", sum(1 for v in values if v == ""))
    return c


def _split_first(values: list[str | None]) -> Counter:
    c: Counter = Counter()
    for v in values:
        c[(v or "<缺失>").split("/")[0]] += 1
    return c


def _peiti_by_group(records: list[dict]) -> dict:
    from video_annotator import taxonomy as T

    out: dict[str, Counter] = {"场景培体": Counter(), "人物培体": Counter()}
    empty = 0
    for r in records:
        v = (r.get("labels") or {}).get("培体")
        if v is None:
            continue
        if v == "":
            empty += 1
            continue
        for name in v.split("_"):
            group = T.peiti_group(name)
            if "场景培体" in group:
                out["场景培体"][name] += 1
            if "人物培体" in group:
                out["人物培体"][name] += 1
    result = {g: dict(c.most_common()) for g, c in out.items()}
    if empty:
        result["<空>"] = {"计数": empty}  # type: ignore[assignment]
    return result


def generate_report(annotations_path: Path, out_dir: Path) -> Path:
    records = load_records(annotations_path)
    n = len(records)
    by_status = Counter(r.get("status") for r in records)
    latencies = [r["latency_s"] for r in records if isinstance(r.get("latency_s"), (int, float))]

    scene = [(r.get("labels") or {}).get("场景类别") for r in records]
    subject = [(r.get("labels") or {}).get("被拍摄主体") for r in records]
    feature = [(r.get("labels") or {}).get("场景特征") for r in records]

    # 运镜：top1（第一位推荐）与全部位次合计；附运镜大类
    top1: Counter = Counter()
    any_pos: Counter = Counter()
    for r in records:
        mv = r.get("movement_top5") or []
        if mv:
            top1[mv[0]["name"]] += 1
        for t in mv:
            any_pos[t["name"]] += 1

    issues_preview = [
        {"id": r.get("id"), "issues": r.get("issues"), "status": r.get("status")}
        for r in records
        if r.get("issues") or r.get("status") == "error"
    ][:80]

    report = {
        "model": next((r.get("model") for r in records if r.get("model")), None),
        "n": n,
        "status": dict(by_status),
        "latency_s": {
            "avg": round(sum(latencies) / len(latencies), 1) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
        "dims": {
            "场景大类": dict(_split_first(scene).most_common()),
            "场景小类": dict(_split_last(scene, split=True).most_common()),
            "主体大类": dict(_split_first(subject).most_common()),
            "主体小类": dict(_split_last(subject, split=True).most_common()),
            "场景特征": dict(_split_first(feature).most_common()),
            "培体": _peiti_by_group(records),
            "运镜top1": dict(top1.most_common()),
            "运镜全位次": dict(any_pos.most_common()),
        },
        "问题清单（前80条）": issues_preview,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # summary.csv（utf-8-sig 方便 Excel 直接打开）
    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "场景大类", "场景小类", "被拍摄主体", "培体", "场景特征",
                    "运镜1", "运镜2", "运镜3", "运镜4", "运镜5", "status"])
        for r in records:
            labels = r.get("labels") or {}
            scene_v = labels.get("场景类别") or ""
            major, _, minor = scene_v.partition("/")
            mv = [t["name"] for t in (r.get("movement_top5") or [])]
            mv += [""] * (5 - len(mv))
            w.writerow([r.get("id"), major, minor.replace("_", "|"),
                        labels.get("被拍摄主体") or "", labels.get("培体") or "",
                        labels.get("场景特征") or "", *mv, r.get("status")])
    return report_path
