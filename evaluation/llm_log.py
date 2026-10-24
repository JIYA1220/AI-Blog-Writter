"""
Logging wrapper: records every prompt and raw response verbatim as JSON lines.
"""

import datetime
import json
import os
import time
from typing import Optional

from utils.json_utils import message_text


class LoggedLLM:
    """Wraps a chat model; every ainvoke is appended to `log_path` (prompt, raw response, timing)."""

    def __init__(self, inner, log_path: str, model: str, temperature: float, tag: str = ""):
        self.inner, self.log_path = inner, log_path
        self.model, self.temperature, self.tag = model, temperature, tag

    async def ainvoke(self, prompt):
        t0 = time.time()
        error: Optional[str] = None
        content = None
        try:
            response = await self.inner.ainvoke(prompt)
            content = message_text(response.content)
            return response
        except Exception as e:                       # logged, then re-raised
            error = repr(e)
            raise
        finally:
            record = {"ts": datetime.datetime.now().isoformat(timespec="seconds"),
                      "tag": self.tag, "model": self.model, "temperature": self.temperature,
                      "seconds": round(time.time() - t0, 2),
                      "prompt": prompt, "response": content, "error": error}
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


def make_run_dir(root: str, name: str) -> str:
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(root, f"{name}-{stamp}")
    os.makedirs(path, exist_ok=True)
    return path
