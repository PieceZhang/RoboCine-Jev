"""单张图片标注：prompt 组装 -> VLM 调用 -> 校验夹紧 -> 带反馈重试。

所有取值严格来自 taxonomy.py 词表；VLM 输出经 sanitize 夹紧（越界剔除、
大类推断、top5 过滤去重、培体归一），不合格先带反馈重试一次，仍不合格走
确定性兜底并在 issues 标记，保证单条记录字段完整、永不落空。
"""
from __future__ import annotations

import datetime
import time
from pathlib import Path
from typing import Any, Optional

from video_annotator import taxonomy as T
from video_annotator.llm import extract_json

SYSTEM = "你是专业的画面标注与运镜推荐助手。只输出一个 JSON 对象，不要解释，不要输出其他文字。"

FALLBACK_TOP5 = ["前推运镜", "环绕运镜", "主角跟随", "横移运镜", "静止"]


def _as_list(x: Any) -> list:
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def build_prompt(scene_feature: bool = True, peiti: bool = True) -> str:
    lines: list[str] = ["任务：对一张图片做标注，并推荐最适合的运镜。", ""]
    lines.append("【候选词表】所有取值必须严格来自下表，不得臆造：")
    lines.append("1. 场景大类（单选）：" + " / ".join(T.SCENE_MAJOR))
    lines.append("2. 场景小类（多选 1-3 个，必须属于所选场景大类）：")
    for major in T.SCENE_MAJOR:
        opts = "｜".join(f"{n}（{d}）" for n, d in T.SCENE_MINOR[major])
        lines.append(f"   - {major}：{opts}")
    lines.append("3. 主体大类（单选）：" + " / ".join(T.SUBJECT_MAJOR))
    lines.append("4. 主体小类（单选，必须属于所选主体大类）：")
    for major in T.SUBJECT_MAJOR:
        opts = "｜".join(f"{n}（{d}）" for n, d in T.SUBJECT_MINOR[major])
        lines.append(f"   - {major}：{opts}")
    n = 5
    if peiti:
        lines.append(f"{n}. 培体（多选 0-3 个；画面中的陪衬物体；没有合适的就给空数组 []）：")
        lines.append("   - 场景培体：" + "｜".join(f"{x}（{d}）" for x, d in T.PEITI_SCENE))
        lines.append("   - 人物培体：" + "｜".join(f"{x}（{d}）" for x, d in T.PEITI_PERSON))
        n += 1
    if scene_feature:
        lines.append(
            f"{n}. 场景特征（单选）："
            + "｜".join(f"{name}（{d}）" for name, d in T.SCENE_FEATURES)
        )
        n += 1
    lines.append(
        f"{n}. 运镜推荐 top5：从下列 22 项中，按「拍摄这类画面最合适的程度」从高到低给出 5 项"
        "（基础运镜：前推运镜/后拉运镜/横摇/竖摇/快速摇摄/横移运镜/竖移运镜/环绕运镜/环视运镜/主角跟随；"
        "电影运镜：希区柯克/盗梦空间/低角度运镜/重庆森林/晃动镜头/倾斜镜头/过肩镜头/第一视POV；"
        "网感运镜：格莱美运镜/动感跟随/百万运镜；"
        "无：静止）："
    )
    lines.append("   " + "｜".join(f"{name}（{d}）" for name, d in T.MOVEMENTS))
    lines += [
        "",
        "【规则】",
        "- 只依据画面内容判断；无人物且无明确物体主体时判为 空镜/纯场景",
        "- 场景小类、主体小类必须属于各自所选大类；没有完美选项时选列表中最接近的一项，"
        "宁可保守，不要臆造词表外的值",
        "- 运镜 top5 必须来自 22 项列表，按适配度排序，每项理由不超过 15 字",
        "",
        "【输出】只输出一个 JSON 对象：",
        '{"场景大类":"...","场景小类":["..."],"主体大类":"...","主体小类":"...",'
        + ('"培体":["..."],' if peiti else "")
        + ('"场景特征":"...",' if scene_feature else "")
        + '"运镜top5":[{"name":"...","reason":"..."},{"name":"...","reason":"..."},...]}',
        "",
        "示例（内容仅供格式参考）：",
        '{"场景大类":"室内","场景小类":["咖啡厅、花店"],"主体大类":"单人","主体小类":"中青年女性",'
        + ('"培体":["杯子","桌子"],' if peiti else "")
        + ('"场景特征":"宽敞半开放或封闭空间",' if scene_feature else "")
        + '"运镜top5":[{"name":"环绕运镜","reason":"主体居中可绕拍"},'
        '{"name":"前推运镜","reason":"突出人物层次"},'
        '{"name":"主角跟随","reason":"人物若移动可跟拍"},'
        '{"name":"横移运镜","reason":"展示环境横向层次"},'
        '{"name":"静止","reason":"静态构图稳出新"}]}',
        "",
        "只输出 JSON。",
    ]
    return "\n".join(lines)


def sanitize(obj: dict, *, scene_feature: bool = True, peiti: bool = True) -> tuple[dict, list[dict], list[str]]:
    """把 VLM 输出夹紧到词表内。

    返回 (labels, top5, issues)。labels 各字段可能为 None/空（调用方据
    need_retry 判断是否重试）；确定性兜底在 finalize 里做。
    """
    issues: list[str] = []

    # ---- 场景大类 / 小类
    major = T.canonical(str(obj.get("场景大类", "")).strip())
    if major not in T.SCENE_MAJOR:
        inferred = None
        for m, names in T.SCENE_MINOR.items():
            opts = {n for n, _ in names}
            if any(T.canonical(str(x)) in opts for x in _as_list(obj.get("场景小类"))):
                inferred = m
                break
        if inferred:
            major = inferred
            issues.append(f"场景大类无效，按小类推断为「{major}」")
        else:
            major = None
            issues.append(f"场景大类无效：{obj.get('场景大类')!r}")

    minors: list[str] = []
    if major:
        opts = set(T.scene_minor_options(major))
        for x in _as_list(obj.get("场景小类")):
            c = T.canonical(str(x).strip())
            if c in opts:
                if c not in minors:
                    minors.append(c)
            else:
                issues.append(f"场景小类越界剔除：{x!r}")
    if major and not minors:
        issues.append("场景小类为空")

    if len(minors) > 3:
        issues.append("场景小类超过 3 项，截断")
        minors = minors[:3]

    # ---- 主体大类 / 小类
    smajor = T.canonical(str(obj.get("主体大类", "")).strip())
    if smajor not in T.SUBJECT_MAJOR:
        c = T.canonical(str(obj.get("主体小类", "")).strip())
        smajor = next(
            (m for m, names in T.SUBJECT_MINOR.items() if c in {n for n, _ in names}), None
        )
        if smajor:
            issues.append(f"主体大类无效，按小类推断为「{smajor}」")
        else:
            issues.append(f"主体大类无效：{obj.get('主体大类')!r}")

    sminor = T.canonical(str(obj.get("主体小类", "")).strip())
    if smajor:
        opts = T.subject_minor_options(smajor)
        if sminor not in opts:
            issues.append(f"主体小类与主体大类不符：{sminor!r}")
            sminor = None

    # ---- 培体（多选 0-3，空合法）
    peiti_vals: list[str] = []
    if peiti:
        valid_peiti = set(T.peiti_names())
        for x in _as_list(obj.get("培体")):
            c = T.canonical(str(x).strip())
            if c in valid_peiti:
                if c not in peiti_vals:
                    peiti_vals.append(c)
            elif str(x).strip():
                issues.append(f"培体词表外剔除：{x!r}")
        peiti_vals = peiti_vals[:3]

    # ---- 场景特征
    feature = None
    if scene_feature:
        feature = T.canonical(str(obj.get("场景特征", "")).strip())
        if feature not in {n for n, _ in T.SCENE_FEATURES}:
            issues.append(f"场景特征无效：{feature!r}")
            feature = None

    # ---- 运镜 top5
    valid = T.movement_names()
    top5: list[dict] = []
    for item in _as_list(obj.get("运镜top5") or obj.get("movement_top5")):
        if isinstance(item, dict):
            raw_name = str(item.get("name") or item.get("运镜") or "").strip()
            reason = str(item.get("reason", "")).strip()
        else:
            raw_name, reason = str(item).strip(), ""
        c = T.canonical(raw_name)
        if c not in valid:
            hits = [n for n in valid if n and (n in c or c in n)]
            c = hits[0] if len(hits) == 1 else None
        if c is None:
            if raw_name:
                issues.append(f"运镜词表外剔除：{raw_name!r}")
            continue
        if all(t["name"] != c for t in top5):
            top5.append({"name": c, "atomic": T.MOVEMENT_ATOMIC.get(c), "reason": reason[:20]})
    top5 = top5[:5]
    if len(top5) < 5:
        issues.append(f"运镜 top5 不足 5 项（{len(top5)}）")

    labels: dict = {}
    labels["场景类别"] = (
        major + ("/" + "_".join(minors) if minors else "") if major else None
    )
    labels["被拍摄主体"] = (
        f"{smajor}/{sminor}" if smajor and sminor else (smajor if smajor else None)
    )
    if peiti:
        labels["培体"] = "_".join(peiti_vals) if peiti_vals else ""
    if scene_feature:
        labels["场景特征"] = feature
    return labels, top5, issues


def need_retry(labels: dict, top5: list[dict]) -> bool:
    return (
        labels.get("场景类别") is None
        or "/" not in (labels.get("场景类别") or "")
        or labels.get("被拍摄主体") is None
        or "/" not in (labels.get("被拍摄主体") or "")
        or len(top5) < 5
        or ("场景特征" in labels and labels["场景特征"] is None)
    )


def _feedback_prompt(issues: list[str], scene_feature: bool, peiti: bool) -> str:
    return (
        "注意：上一次输出无法通过校验，问题如下：\n- "
        + "\n- ".join(issues[:8])
        + "\n请严格按候选词表重新输出完整 JSON（所有小类必须属于所选大类，运镜 top5 给满 5 项）。\n"
        + "只输出一个 JSON 对象，以 { 开头、} 结尾。词表：\n"
        + build_prompt(scene_feature, peiti)
    )


def finalize(labels: dict, top5: list[dict], issues: list[str],
             scene_feature: bool, peiti: bool) -> tuple[dict, list[dict], bool]:
    """确定性兜底：缺什么补什么，保证字段完整、永不落空。"""
    fallback = False
    if not labels.get("场景类别"):
        labels["场景类别"] = "室内"
        issues.append("兜底：场景类别=室内")
        fallback = True
    if not labels.get("被拍摄主体"):
        labels["被拍摄主体"] = "空镜/纯场景"
        issues.append("兜底：被拍摄主体=空镜/纯场景")
        fallback = True
    if peiti and labels.get("培体") is None:
        labels["培体"] = ""
        fallback = True
    if scene_feature and not labels.get("场景特征"):
        labels["场景特征"] = "宽敞半开放或封闭空间"
        issues.append("兜底：场景特征")
        fallback = True
    names = [t["name"] for t in top5]
    for name in FALLBACK_TOP5:
        if len(top5) >= 5:
            break
        if name not in names:
            top5.append({"name": name, "atomic": T.MOVEMENT_ATOMIC.get(name), "reason": "兜底候选"})
            fallback = True
    return labels, top5, fallback


def annotate_one(
    client,
    image_path: Path,
    *,
    record_id: Optional[str] = None,
    scene_feature: bool = True,
    peiti: bool = True,
    keep_raw: bool = False,
    max_tokens: int = 900,
    temperature: float = 0.1,
) -> dict:
    """标注单张图片，返回一条记录（失败记 status=error，不抛致命错误）。

    record_id 默认取文件名；批量跑不同子目录时建议传相对路径避免同名冲突。
    """
    record: dict = {
        "id": record_id or image_path.name,
        "path": str(image_path),
        "type": "image",
        "model": None,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "labels": {},
        "movement_top5": [],
        "issues": [],
        "status": "ok",
        "error": None,
        "latency_s": None,
    }
    t0 = time.time()
    prompt = build_prompt(scene_feature, peiti)
    try:
        best = None
        for attempt in range(2):
            reply = client.vision(prompt, [image_path], system=SYSTEM,
                                  max_tokens=max_tokens, temperature=temperature)
            obj = extract_json(reply)
            if obj is None:
                prompt = "上一次无法解析，请只输出完整 JSON。\n" + build_prompt(scene_feature, peiti)
                continue
            labels, top5, issues = sanitize(obj, scene_feature=scene_feature, peiti=peiti)
            # Completeness takes precedence over the number of cosmetic corrections.
            score = (need_retry(labels, top5), len(issues))
            if best is None or score < best[0]:
                best = (score, obj, labels, top5, issues)
            if not need_retry(labels, top5) and not issues:
                break
            prompt = _feedback_prompt(issues, scene_feature, peiti)
        if best is None:
            raise ValueError("两次输出均无法解析出 JSON")
        _, obj, labels, top5, issues = best

        labels, top5, fallback = finalize(labels, top5, issues, scene_feature, peiti)
        record["labels"] = labels
        record["movement_top5"] = top5
        record["issues"] = issues
        record["status"] = "fallback" if (fallback or issues) else "ok"
        if keep_raw:
            record["raw"] = obj
    except Exception as e:  # 单条失败不影响整批
        record["status"] = "error"
        record["error"] = f"{type(e).__name__}: {e}"
    try:
        record["model"] = getattr(client, "model", None)
    except Exception:
        record["model"] = None
    record["latency_s"] = round(time.time() - t0, 1)
    return record
