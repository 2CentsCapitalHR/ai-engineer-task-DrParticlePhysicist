import json
from LLM_client import get_LLM


def safe_json_loads(data, fallback=None):
    try:
        if isinstance(data, str):
            return json.loads(data)
        return data
    except:
        return fallback if fallback is not None else data


def detect_red_flags_with_process(process_name, doc_text, filename, kb_chunks):
    prompt = f"""
You are an ADGM compliance expert.

Identified process: {process_name}

Relevant KB context:
{json.dumps(kb_chunks, indent=2, ensure_ascii=False)}

Document: {filename}
Content:
{doc_text}

Task:
Identify any compliance issues or "red flags" in the document.
Return ONLY JSON list:
[{{"issue": "<short>", "explanation": "<detailed>", "match_text": "<exact_text_to_match>"}}]
"""
    resp = get_LLM().invoke(prompt)
    return safe_json_loads(resp, fallback=[])
