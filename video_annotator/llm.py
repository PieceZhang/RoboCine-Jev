"""LM Studio 客户端（OpenAI 兼容 /v1/chat/completions）。

沿用 vlog_agent/llm.py（feat/vlog-agent@13bc617）的模式：/v1/models 自动发现
模型、PIL 缩图 base64 data URL、extract_json 容错解析（小模型常把 JSON 包在
散文或代码围栏里）。差异：模型发现优先选 qwen3-vl 系（32b 优先），请求级
重试，缩图默认 1024px（32B 上下文充裕）。
"""
from __future__ import annotations

import base64
import io
import json
import re
import time
from pathlib import Path
from typing import Any, Optional

import requests
from PIL import Image, ImageOps

DEFAULT_API_URL = "http://127.0.0.1:1234/v1/chat/completions"


def extract_json(text: str) -> Optional[dict]:
    """从 LLM 回复里拉出第一个完整的 JSON 对象。

    兼容 ```json 围栏、散文包裹、字符串值里的大括号；找不到返回 None。
    """
    candidates: list[str] = []
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1))
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : i + 1])
                    break
        start = text.find("{", start + 1)
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def image_data_url(path: Path, max_px: int = 1024, quality: int = 88) -> str:
    """把本地图片缩放成 base64 JPEG data URL。"""
    with Image.open(path) as source:
        img = ImageOps.exif_transpose(source).convert("RGB")
    img.thumbnail((max_px, max_px))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _prefer_qwen3vl(ids: list[str]) -> str:
    """qwen3-vl 系优先（32b > 30b > 其它），否则取第一个。"""
    qwen = [i for i in ids if "qwen3-vl" in i.lower()]
    for tag in ("32b", "30b"):
        hit = [i for i in qwen if tag in i.lower()]
        if hit:
            return hit[0]
    if qwen:
        return qwen[0]
    return ids[0]


class LMSClient:
    """运行中 LM Studio 服务的最小 chat/vision 客户端。"""

    def __init__(
        self,
        api_url: str = DEFAULT_API_URL,
        model: Optional[str] = None,
        timeout: float = 300.0,
        max_px: int = 1024,
        retries: int = 3,
    ):
        self.api_url = api_url
        self.timeout = timeout
        self.max_px = max_px
        self.retries = retries
        self._model = model
        self.session = requests.Session()

    @property
    def model(self) -> str:
        if self._model is None:
            self._model = self.detect_model()
        return self._model

    def detect_model(self) -> str:
        """挑服务里最合适的对话模型：qwen3-vl 系（32b 优先）> 首个非 embedding。"""
        base = self.api_url.rsplit("/v1/", 1)[0]
        r = self.session.get(f"{base}/v1/models", timeout=10)
        r.raise_for_status()
        ids = [m["id"] for m in r.json().get("data", []) if "embed" not in m["id"].lower()]
        if not ids:
            raise RuntimeError("LM Studio 没有已加载的对话模型（lms ls 查看）")
        return _prefer_qwen3vl(ids)

    def chat(self, messages: list[dict], *, max_tokens: int = 1024, temperature: float = 0.1) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        last_err: Exception | None = None
        for attempt in range(self.retries):
            try:
                r = self.session.post(self.api_url, json=payload, timeout=self.timeout)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"] or ""
            except (requests.RequestException, KeyError, ValueError) as e:
                last_err = e
                if attempt < self.retries - 1:
                    time.sleep(2.0 * (attempt + 1))
        raise RuntimeError(f"LM Studio 请求失败（{self.retries} 次）：{last_err}")

    def vision(
        self,
        prompt: str,
        image_paths: list[Path | str],
        *,
        system: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> str:
        content: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": image_data_url(Path(p), self.max_px)}}
            for p in image_paths
        ]
        content.append({"type": "text", "text": prompt})
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})
        return self.chat(messages, max_tokens=max_tokens, temperature=temperature)
