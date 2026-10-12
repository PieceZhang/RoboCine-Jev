"""User Prompt Translator (paper §3.2 + appendix E): LLM decomposes the motion
prompt into a validated 1-3-1 atomic-command plan, grounded in the scene graph.

Backends:
  * TrajPlannerClient — LM Studio (OpenAI-compatible /v1/chat/completions),
    reusing the repo's annotate_vlm calling pattern + retry/backoff and a
    validator-feedback repair round (upgrades over the annotate_vlm client).
  * MockPlanner — deterministic keyword planner for tests / offline runs.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cinema_traj.common import constants as C
from cinema_traj.common.scene import Scene, load_scene
from cinema_traj.core.plan_schema import normalize_plan, validate_plan
from vlm_intention.common.atomic_io import atomic_write_json, load_json
from vlm_intention.common.config import load_config

# --------------------------------------------------------------------------- #
# Prompt design (paper appendix E: toolset + prefs + pattern + examples)
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """你是一个电影级运镜规划器（cinematographic trajectory planner）。
给定 3D 场景图（房间/物体 OBB/贴墙挂顶关系/门洞）和可选的动态主体（subject）轨迹，
把用户的自然语言运镜请求分解为原子运镜指令序列（JSON）。

## 三层工具集
1. anchor 锚点选择器 A：为每个目标确定初始视角。需要给出
   - elevation ∈ {low(10-20°), medium(25-35°), high(40-55°), overhead(60-80°)}
   - distance ∈ {close(×0.7), medium(×1.0), far(×1.4)}
2. 物体级运镜 T_o（object_traj，13 种）：
   orbit_full / orbit_half / orbit_quarter（环绕 360°/180°/90°）、
   pan_left / pan_right（定点水平摇镜）、tilt_up / tilt_down（定点俯仰）、
   move_in / move_out（推近 / 拉远）、crane（升降弧线）、
   zoom_in_out / zoom_out_in（变焦并回位）、static（定点静止）
3. 主体运镜（subject_traj，动态主体跟拍）：
   track_follow（保持固定相对方位跟随）、track_orbit（跟随中环绕）、
   track_lead（过肩/前瞻机位，lead_angle≈±165° 为过肩）
4. 过渡运镜 T_t（transition，arc）：带曲率 alpha_deg ∈ [-89°, 89°] 的弧线
   （0 = 直线，正值向左弯，负值向右弯）。

## 硬约束（违反即无效）
- 1-3-1 模式：N 个目标 → 恰好 3N-1 步：
  A₁ → T_o,1 → (A₂ → T_t → T_o,2)* → … → A_N → T_t → T_o,N
  即：每个目标一个 anchor + 一个运镜；相邻目标之间恰好一个 arc 过渡。
- placement 约束：placement=wall 的物体禁用 orbit_full / orbit_half；
  ceiling 禁用 crane；floor_wall 禁用 orbit_full。
- 用户未指定顺序时用空间上合理的最近邻顺序，并把 user_specified_order 置 false。
- 场景中没有 subject 轨迹时不得使用 subject_traj 或 target="subject"。

## 输出格式（只返回 JSON，不要 markdown 代码块）
{
  "reasoning": "简短中文推理",
  "user_specified_order": true,
  "object_sequence": ["sofa_01", "table_01", "subject"],
  "steps": [
    {"step": 1, "kind": "anchor", "target": "sofa_01",
     "elevation": "medium", "distance": "medium"},
    {"step": 2, "kind": "object_traj", "target": "sofa_01",
     "type": "orbit_half", "duration_s": 6.0},
    {"step": 3, "kind": "anchor", "target": "table_01",
     "elevation": "high", "distance": "far"},
    {"step": 4, "kind": "transition", "target": "table_01",
     "type": "arc", "alpha_deg": 0.0, "duration_s": 3.0},
    {"step": 5, "kind": "object_traj", "target": "table_01",
     "type": "orbit_quarter", "duration_s": 5.0},
    {"step": 6, "kind": "anchor", "target": "subject",
     "elevation": "medium", "distance": "medium"},
    {"step": 7, "kind": "transition", "target": "subject",
     "type": "arc", "alpha_deg": 0.0, "duration_s": 3.0},
    {"step": 8, "kind": "subject_traj", "target": "subject",
     "type": "track_lead", "duration_s": 8.0}
  ]
}"""


# --------------------------------------------------------------------------- #
# LM Studio client
# --------------------------------------------------------------------------- #
class TrajPlannerClient:
    """OpenAI-compatible chat client (LM Studio), with retries and one
    validator-feedback repair round."""

    def __init__(self, cfg=None):
        cfg = cfg or load_config()
        pc = cfg.get("cinema_traj", {}).get("planner", {}) if isinstance(cfg, dict) else {}
        vlm = cfg.get("models", {}).get("vlm", {}) if isinstance(cfg, dict) else {}

        def pick(key, default):
            return pc.get(key, vlm.get(key, default))

        self.api_url = pick("api_url", "http://127.0.0.1:1234/v1/chat/completions")
        self.model_name = pick("model_name", "qwen/qwen3-vl-30b")
        self.temperature = float(pick("temperature", 0.1))
        self.max_tokens = int(pick("max_tokens", 4096))
        self.timeout = float(pick("request_timeout_s", 180))
        self.retries = int(pc.get("retries", 3))

    # -- transport --------------------------------------------------------- #
    def _chat(self, messages: list) -> str:
        import requests

        payload = {"model": self.model_name, "messages": messages,
                   "temperature": self.temperature, "max_tokens": self.max_tokens}
        last_err = None
        for attempt in range(self.retries):
            try:
                r = requests.post(self.api_url, json=payload, timeout=self.timeout)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            except Exception as e:                        # noqa: BLE001 — retry any failure
                last_err = e
                if attempt < self.retries - 1:
                    time.sleep(2.0 * (2 ** attempt))
        raise RuntimeError(f"LLM request failed after {self.retries} tries: {last_err}")

    @staticmethod
    def _parse_json(text: str) -> dict:
        t = text.strip()
        if t.startswith("```"):
            t = t.strip("`")
            if t.lower().startswith("json"):
                t = t[4:]
        try:
            obj = json.loads(t)
            if isinstance(obj, dict):
                return obj
            # tolerate "```json\n{...}\n```" stripping artifacts
        except json.JSONDecodeError:
            start, end = t.find("{"), t.rfind("}")
            if start >= 0 and end > start:
                return json.loads(t[start:end + 1])
            raise
        raise ValueError("LLM output is not a JSON object")

    # -- planning ----------------------------------------------------------- #
    def plan(self, prompt: str, scene: Scene) -> dict:
        user_msg = (f"## 场景图\n{scene.summary()}\n\n"
                    f"## 用户请求\n{prompt}\n\n"
                    f"请输出 1-3-1 原子运镜计划 JSON（只返回 JSON）。")
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg}]
        raw = self._chat(messages)
        plan = normalize_plan({**self._parse_json(raw), "prompt": prompt,
                               "model": self.model_name}, scene)
        errs = validate_plan(plan, scene)
        if errs:                                          # one repair round with feedback
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content":
                             "你的计划未通过校验，错误如下：\n- " + "\n- ".join(errs) +
                             "\n请修正后重新输出完整 JSON。"})
            raw = self._chat(messages)
            plan = normalize_plan({**self._parse_json(raw), "prompt": prompt,
                                   "model": self.model_name}, scene)
            errs = validate_plan(plan, scene)
        if errs:
            raise ValueError("plan validation failed even after repair:\n- " + "\n- ".join(errs))
        return plan


# --------------------------------------------------------------------------- #
# MockPlanner — deterministic keyword rules (tests / offline)
# --------------------------------------------------------------------------- #
_MOVEMENT_RULES = [
    (("track_lead", "subject_traj"), ("过肩", "over-the-shoulder", "前瞻机位")),
    (("track_orbit", "subject_traj"), ("环绕跟随", "边环绕边跟", "跟拍环绕", "orbit.*follow")),
    (("track_follow", "subject_traj"), ("跟拍", "跟随", "跟踪", "follow", "track")),
    (("orbit_full", "object_traj"), ("360", "全周", "整圈", "full orbit")),
    (("orbit_half", "object_traj"), ("半圈", "180", "half orbit", "绕半")),
    (("orbit_quarter", "object_traj"), ("四分之一", "90", "quarter", "环绕", "绕一圈", "orbit")),
    (("move_in", "object_traj"), ("推近", "移近", "靠近", "推到", "move in", "dolly in", "zoom in", "变焦推")),
    (("move_out", "object_traj"), ("拉远", "远离", "后退", "move out", "dolly out", "pull back")),
    (("crane", "object_traj"), ("升降", "摇臂", "升起", "crane", "俯冲")),
    (("zoom_in_out", "object_traj"), ("变焦", "推拉变焦", "zoom")),
    (("pan_left", "object_traj"), ("向左摇", "左摇", "pan left")),
    (("pan_right", "object_traj"), ("摇镜", "平移", "pan right", "pan")),
    (("tilt_down", "object_traj"), ("俯视", "下俯", "tilt down")),
    (("tilt_up", "object_traj"), ("仰视", "上仰", "tilt up")),
    (("static", "object_traj"), ("静止", "定机位", "static", "定格")),
]

_ELEV_RULES = [("overhead", ("俯视", "顶视", "overhead")), ("high", ("高", "high")),
               ("low", ("低", "low")), ("medium", ())]
_DIST_RULES = [("close", ("近景", "特写", "close")), ("far", ("远景", "far")),
               ("medium", ())]


def _first_match(prompt: str, rules):
    for value, kws in rules:
        if any(k.lower() in prompt.lower() for k in kws):
            return value
    return None


def _find_objects(prompt: str, scene: Scene) -> list:
    """Mentioned objects (by id or label substring), in order of appearance;
    the subject counts when a tracking keyword appears."""
    hits = []
    for o in scene.objects:
        keys = [o.id.lower(), o.label.lower()] + \
               [w for w in o.label.replace("(", " ").split() if len(w) >= 2]
        for k in dict.fromkeys(keys):
            idx = prompt.lower().find(k.lower())
            if idx >= 0:
                hits.append((idx, o))
                break
    hits.sort(key=lambda x: x[0])
    objs = [o.id for _, o in hits]
    if scene.subject is not None and _first_match(prompt, _MOVEMENT_RULES[:3]):
        low = prompt.lower()
        idx = min((low.find(x) for x in ("演员", "主体", "假人") if x in low),
                  default=-1)
        if idx >= 0 or not objs:
            objs = objs + [scene.subject.subject_id]
    return objs


class MockPlanner:
    """Deterministic keyword planner — output always satisfies the validator."""

    def __init__(self, cfg=None):
        pass

    def plan(self, prompt: str, scene: Scene) -> dict:
        targets = _find_objects(prompt, scene)
        if not targets:                                  # default: first object, static
            targets = [scene.objects[0].id] if scene.objects else []
        subject_id = scene.subject.subject_id if scene.subject else "subject"
        steps, n = [], 0
        for tgt in targets:
            is_subject = tgt == subject_id
            elev = _first_match(prompt, _ELEV_RULES) or "medium"
            dist = _first_match(prompt, _DIST_RULES) or "medium"
            n += 1
            steps.append({"step": len(steps) + 1, "kind": "anchor", "target": tgt,
                          "elevation": elev, "distance": dist})
            if n > 1:                                  # paper: A → T_t → T_o
                steps.append({"step": len(steps) + 1, "kind": "transition", "target": tgt,
                              "type": "arc", "alpha_deg": 0.0, "duration_s": 3.0})
            movement, kind = self._subject_movement(prompt) if is_subject else \
                self._object_movement(prompt, scene, tgt)
            steps.append({"step": len(steps) + 1, "kind": kind, "target": tgt,
                          "type": movement, "duration_s": 8.0 if is_subject else 6.0})
        return {"plan_id": "mock", "prompt": prompt, "model": "mock-planner",
                "user_specified_order": True, "object_sequence": list(targets),
                "steps": steps, "reasoning": "mock keyword planner"}

    @staticmethod
    def _subject_movement(prompt: str) -> tuple:
        if any(k in prompt for k in ("过肩", "over-the-shoulder", "前瞻")):
            return "track_lead", "subject_traj"
        if any(k in prompt for k in ("环绕跟随", "边环绕边跟")):
            return "track_orbit", "subject_traj"
        return "track_follow", "subject_traj"

    @staticmethod
    def _object_movement(prompt: str, scene: Scene, tgt: str) -> tuple:
        obj = scene.object_by_id(tgt)
        forbidden = C.PLACEMENT_FORBIDDEN.get(obj.placement, set()) if obj else set()
        for (movement, kind), kws in _MOVEMENT_RULES:
            if kind != "object_traj" or movement in forbidden:
                continue
            if any(k.lower() in prompt.lower() for k in kws):
                return movement, kind
        return "orbit_quarter", "object_traj"            # safe default for all placements


# --------------------------------------------------------------------------- #
# Stage entry
# --------------------------------------------------------------------------- #
def run(cfg=None, scene_path=None, prompt=None, prompt_file=None,
        mock=False, force=False) -> dict:
    cfg = cfg or load_config()
    ct = cfg.get("cinema_traj", {})
    scene_path = scene_path or ct.get("scene", "")
    scene = load_scene(scene_path)
    out_dir = _out_dir(cfg, scene)
    out_path = out_dir / "plan.json"

    if prompt is None and prompt_file:
        prompt = Path(prompt_file).read_text(encoding="utf-8").strip()
    prompt = prompt or ct.get("prompt", "")

    if out_path.exists() and not force:
        cached = load_json(out_path)
        if cached and cached.get("prompt") == prompt:
            print(f"↩️  复用已有 plan: {out_path}")
            return cached

    planner = MockPlanner(cfg) if mock else TrajPlannerClient(cfg)
    backend = "mock" if mock else planner.model_name
    print(f"🧠 运镜规划（{backend}）: {prompt[:60]}…")
    plan = planner.plan(prompt, scene)

    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_path, plan)
    print(f"✅ plan → {out_path}（{len(plan['steps'])} 步，目标 {plan['object_sequence']}）")
    return plan


def _out_dir(cfg, scene) -> Path:
    ct = cfg.get("cinema_traj", {})
    base = ct.get("outputs_dir", "cinema_traj_out")
    p = Path(base)
    if not p.is_absolute():
        from vlm_intention.common.config import repo_root
        p = repo_root() / p
    return p / scene.scene_id


if __name__ == "__main__":                                # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="LLM 运镜规划（User Prompt Translator）")
    ap.add_argument("--scene", help="场景 JSON 路径")
    ap.add_argument("--prompt", help="运镜请求文本")
    ap.add_argument("--prompt-file", help="运镜请求文件")
    ap.add_argument("--mock", action="store_true", help="用确定性 MockPlanner（离线）")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--config", help="自定义 config.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config) if a.config else None
    run(cfg, scene_path=a.scene, prompt=a.prompt, prompt_file=a.prompt_file,
        mock=a.mock, force=a.force)
