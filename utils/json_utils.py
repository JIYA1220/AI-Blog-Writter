# utils/json_utils.py
# Shared helper for turning an LLM reply into a JSON object.

import json


def message_text(content) -> str:
    """
    Plain text of a chat reply. Most models return a string; newer Gemini models return a
    list of content blocks such as [{"type": "text", "text": "..."}, ...].
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type", "text") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return "" if content is None else str(content)


def parse_llm_json(raw):
    """
    Parse JSON from an LLM reply. Accepts plain JSON, JSON wrapped in a
    ``` or ```json fence, and a leading bare 'json' label.
    Raises json.JSONDecodeError if the text is not valid JSON.
    """
    text = message_text(raw).strip()

    if text.startswith("```"):
        lines = text.splitlines()
        # drop the opening fence line, and the closing fence if present
        body = lines[1:-1] if len(lines) > 1 and lines[-1].strip().startswith("```") else lines[1:]
        text = "\n".join(body).strip()

    if text.startswith("json"):
        text = text[4:].strip()

    # strict=False: small models often leave raw newlines/tabs inside JSON strings
    return json.loads(text, strict=False)
