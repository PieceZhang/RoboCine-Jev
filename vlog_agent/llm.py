"""LM Studio client for the vlog agent (OpenAI-compatible /v1/chat/completions).

Same endpoint the rest of the repo already uses (cinema_traj/stages/plan.py,
vlm_intention/stages/annotate_vlm.py). Adds: model autodetect via /v1/models,
vision messages from local image files, and tolerant JSON extraction — small
local models frequently wrap JSON in prose or code fences.
"""
from __future__ import annotations

import base64
import io
import json
import re
from pathlib import Path
from typing import Any, Optional

import requests
from PIL import Image

DEFAULT_API_URL = "http://127.0.0.1:1234/v1/chat/completions"


def extract_json(text: str) -> Optional[dict]:
    """Pull the first balanced JSON object out of an LLM reply.

    Handles ```json fences, prose-wrapped objects, and braces inside string
    values. Returns None when nothing parseable is found.
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


def image_data_url(path: Path, max_px: int = 640, quality: int = 85) -> str:
    """Downscale a local image to a base64 JPEG data URL (keeps 2B context small)."""
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_px, max_px))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


class LMSClient:
    """Minimal chat/vision client for a running LM Studio server."""

    def __init__(
        self,
        api_url: str = DEFAULT_API_URL,
        model: Optional[str] = None,
        timeout: float = 180.0,
    ):
        self.api_url = api_url
        self.timeout = timeout
        self._model = model
        self.session = requests.Session()

    @property
    def model(self) -> str:
        if self._model is None:
            self._model = self.detect_model()
        return self._model

    def detect_model(self) -> str:
        """Pick the first non-embedding model the server has loaded."""
        base = self.api_url.rsplit("/v1/", 1)[0]
        r = self.session.get(f"{base}/v1/models", timeout=10)
        r.raise_for_status()
        ids = [m["id"] for m in r.json().get("data", []) if "embed" not in m["id"].lower()]
        if not ids:
            raise RuntimeError("LM Studio has no chat models loaded (lms ls to check)")
        return ids[0]

    def chat(self, messages: list[dict], *, max_tokens: int = 1024, temperature: float = 0.1) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        r = self.session.post(self.api_url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] or ""

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
            {"type": "image_url", "image_url": {"url": image_data_url(Path(p))}}
            for p in image_paths
        ]
        content.append({"type": "text", "text": prompt})
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})
        return self.chat(messages, max_tokens=max_tokens, temperature=temperature)
