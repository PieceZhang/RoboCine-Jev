"""Stage: annotate_vlm — legacy VLM 5-direction tagger (Qwen3-VL-30b via LM Studio).

Migrated from final_annotation.py. Outputs obstacle_annotations_final.json,
now used as a CALIBRATION reference (depth is the main labeler). Upgraded to
atomic writes; config-driven endpoint/model/thumbnail.

Requires a running LM Studio server (OpenAI-compatible) at models.vlm.api_url.
"""
import base64
import io
import json
import time
from pathlib import Path
from typing import Optional

from PIL import Image
import requests

from ..common.atomic_io import atomic_write_json, load_json
from ..common.config import load_config
from ..common.paths import resolve_base, discover_images
from ..common.schema import DIRS, build_annotation_record

PROMPT = """请分析这张图片，判断在以下5个方向是否有障碍物：

**障碍物定义（精确标准）：**
- **前、左、右**：能碰撞到的墙壁、建筑物、固定设施都算障碍物
- **上（头顶）**：0.5米以内主体以外的物体算障碍物（如近的墙体、门框、天花板）
- **下（脚下）**：0.5米以内主体以外的物体算障碍物（如近的桌面、地面）

**不算障碍物：** 画面中的人物（单人/多人）、宠物、手持物品、包包、食物、玩偶、植物等画面主体

**方向定义：**
- 前 (Front)：观察者正前方能碰撞到的墙壁或固定物体
- 左 (Left)：左侧能碰撞到的墙壁或固定物体
- 右 (Right)：右侧能碰撞到的墙壁或固定物体
- 上 (Up/Top)：0.5米范围内的头顶障碍物（近的墙体、门框、天花板）
- 下 (Down/Bottom)：0.5米范围内的脚下障碍物（近的桌面、地面）

请以JSON格式返回结果：
{
  "前": true/false,
  "左": true/false,
  "右": true/false,
  "上": true/false,
  "下": true/false
}

只返回JSON，不要其他解释性文字。"""


class VLMObstacleAnnotator:
    def __init__(self, cfg=None):
        self.cfg = cfg or load_config()
        self.base = resolve_base(self.cfg)
        self.out = self.base / self.cfg.outputs.vlm_annotations
        vc = self.cfg.models.vlm
        self.api_url = vc.api_url
        self.model_name = vc.model_name
        self.temperature = vc.temperature
        self.max_tokens = vc.max_tokens
        self.thumb_px = vc.thumbnail_max_px
        self.timeout = vc.request_timeout_s
        self.existing = load_json(self.out, default={})

    def _encode(self, image_path: Path) -> str:
        with Image.open(image_path) as img:
            if img.width > self.thumb_px or img.height > self.thumb_px:
                img.thumbnail((self.thumb_px, self.thumb_px), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode()

    def _call(self, image_b64: str) -> Optional[dict]:
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ]}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        try:
            r = requests.post(self.api_url, json=payload, timeout=self.timeout)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            ann = json.loads(content.strip())
            return {d: bool(ann[d]) for d in DIRS} if all(d in ann for d in DIRS) else None
        except Exception as e:
            print(f"  - VLM 调用失败: {e}")
            return None

    def annotate_single(self, image_path: Path):
        rel = str(image_path.relative_to(self.base))
        if rel in self.existing:
            return None
        b64 = self._encode(image_path)
        if not b64:
            return None
        ann = self._call(b64)
        if not ann or not all(d in ann for d in DIRS):
            return None
        return build_annotation_record(ann, image_path.name, rel, self.model_name)

    def batch_annotate(self, max_images=None, start_from=0, images=None):
        images = images or discover_images(self.base)
        if max_images:
            images = images[start_from:start_from + max_images]
        else:
            images = images[start_from:]
        print(f"📁 {len(images)} 张 | 模型 {self.model_name} | 输出 {self.out.name}")
        success = skip = err = 0
        t0 = time.time()
        for i, p in enumerate(images):
            if (i + 1) % 10 == 0:
                print(f"  进度 [{i+1}/{len(images)}] | {time.time()-t0:.1f}s")
            try:
                rec = self.annotate_single(p)
                if rec:
                    self.existing[rec["relative_path"]] = rec
                    success += 1
                    if success % 10 == 0:
                        atomic_write_json(self.out, self.existing)
                else:
                    skip += 1
            except Exception as e:
                print(f"  ❌ {p.name}: {e}"); err += 1
        atomic_write_json(self.out, self.existing)
        print(f"🎉 成功 {success} | 跳过 {skip} | 失败 {err} | 总计 {len(self.existing)}")
        self.get_statistics()
        return self.existing

    def get_statistics(self):
        n = len(self.existing)
        if not n:
            return
        print(f"📊 VLM 标注统计（{n} 条）:")
        for d in DIRS:
            c = sum(1 for v in self.existing.values() if v.get(d))
            print(f"  {d}: {c} ({c/n*100:.1f}%)")


def run(cfg=None, max_images=None, start_from=0):
    ann = VLMObstacleAnnotator(cfg)
    ann.batch_annotate(max_images=max_images, start_from=start_from)


if __name__ == "__main__":
    run()
