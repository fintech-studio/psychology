import json
import re
from typing import Optional, Dict, Tuple, List

# Precompiled regex patterns to detect fenced JSON
_CODE_FENCE_JSON_RE = re.compile(
    r"```(?:json)?\s*(\{[\s\S]*\})\s*```",
    re.IGNORECASE,
)


def extract_text_from_response(raw_text: str) -> Tuple[str, bool]:
    if not raw_text:
        return "", False
    try:
        obj = json.loads(raw_text)
    except Exception:
        obj = None

    out = ""
    has_response = False
    if isinstance(obj, dict):
        if 'response' in obj and obj.get('response'):
            out += obj.get('response', '')
            has_response = True
        elif 'content' in obj:
            c = obj['content']
            if isinstance(c, list):
                for item in c:
                    if isinstance(item, dict) and 'text' in item:
                        out += item['text']
                    elif isinstance(item, str):
                        out += item
            elif isinstance(c, str):
                out += c
        elif 'text' in obj:
            out += obj.get('text', '')
        elif 'output' in obj:
            out += obj.get('output', '')
        return out.strip(), has_response

    try:
        for line in raw_text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                out += line + "\n"
                continue
            if isinstance(item, dict):
                if 'response' in item and item.get('response'):
                    out += item.get('response', '')
                    has_response = True
                elif ('content' in item
                      and isinstance(item['content'], str)):
                    out += item['content']
                elif 'text' in item:
                    out += item.get('text', '')
                elif 'thinking' in item:
                    out += item.get('thinking', '')
    except Exception:
        pass
    return out.strip(), has_response


def _find_json_object_in_string(s: str) -> Optional[str]:
    if not s:
        return None
    m = _CODE_FENCE_JSON_RE.search(s)
    if m:
        return m.group(1)
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(s)):
        if s[i] == '{':
            depth += 1
        elif s[i] == '}':
            depth -= 1
            if depth == 0:
                return s[start:i+1]
    return None


def extract_json_from_text(raw_text: str) -> Optional[Dict]:
    if not raw_text:
        return None
    try:
        obj = json.loads(raw_text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = _CODE_FENCE_JSON_RE.search(raw_text)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    candidate = _find_json_object_in_string(raw_text)
    if not candidate:
        return None
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
    except Exception:
        cleaned = candidate.strip().strip('`\n ')
        try:
            obj = json.loads(cleaned)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    return None


def sanitize_text(s: Optional[str]) -> str:
    if not s:
        return ""
    txt = str(s)
    txt = txt.replace('"', '').replace("'", '').replace('*', '')
    txt = "\n".join(
        [line.strip() for line in txt.splitlines() if line.strip()]
    )
    return txt


def as_list(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i).strip() for i in x if str(i).strip()]
    if isinstance(x, str):
        return [line.strip() for line in x.splitlines() if line.strip()]
    return [str(x)]


def parse_advice_json(advice_obj: Dict) -> Dict:
    return {
        'summary': str(advice_obj.get('summary', '')).strip(),
        'recommendations': as_list(
            advice_obj.get('recommendations')
            or advice_obj.get('recommendation')
        ),
        'risk_management': as_list(
            advice_obj.get('risk_management')
            or advice_obj.get('riskManagement')
            or advice_obj.get('risk_managements')
        ),
        'next_steps': as_list(
            advice_obj.get('next_steps') or advice_obj.get('nextSteps')
        ),
        'tone': str(advice_obj.get('tone', '')).strip(),
        'raw': advice_obj,
    }
