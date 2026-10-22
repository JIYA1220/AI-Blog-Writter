# utils/json_utils.py
# Shared helper for turning an LLM reply into a JSON object.

import json


def parse_llm_json(raw: str):
    """
    Parse JSON from an LLM reply. Accepts plain JSON, JSON wrapped in a
    ``` or ```json fence, and a leading bare 'json' label.
    Raises json.JSONDecodeError if the text is not valid JSON.
    """
    text = (raw or "").strip()

    if text.startswith("```"):
        lines = text.splitlines()
        # drop the opening fence line, and the closing fence if present
        body = lines[1:-1] if len(lines) > 1 and lines[-1].strip().startswith("```") else lines[1:]
        text = "\n".join(body).strip()

    if text.startswith("json"):
        text = text[4:].strip()

    return json.loads(text)
