import json
from Vectorstore_utils import search_Vectorstore
from LLM_client import get_LLM


def safe_json_loads(data, fallback=None):
    try:
        if isinstance(data, str):
            return json.loads(data)
        return data
    except Exception:
        return fallback if fallback is not None else data


def fetch_required_docs_llm(process_name, kb_chunks=None, top_k=3):
    if kb_chunks is None:
        kb_chunks_raw = search_Vectorstore(process_name, top_k=top_k)
        kb_chunks = [c["text"] for c in kb_chunks_raw if c.get("text")]
    else:
        kb_chunks = kb_chunks

    prompt = f"""
Process: {process_name}

Relevant Knowledge Base context:
{json.dumps(kb_chunks[:3], indent=2, ensure_ascii=False)}

List ALL documents generally required for this process in ADGM paradigm, purely as per the above context.Make sure that required list containes atleast one entry fom uploaded.
Return ONLY valid JSON:
{{"required": ["DocType1", "DocType2", "..."]}}
"""
    llm_resp = get_LLM().invoke(prompt)
    llm_json = safe_json_loads(llm_resp, fallback={})
    required_list = [
        x.strip() for x in llm_json.get("required", [])
        if x and isinstance(x, str)
    ]
    return required_list


def fetch_missing_docs_llm(uploaded_doc_types, required_list):
    prompt = f"""
Given two lists below, return ONLY the items that are present in the required list but NOT present (case and whitespace insensitive) in the uploaded list. The content of missing list must be sub set of required list.
Do not include anything present in the uploaded list. No duplicates.  And use your reasoning also just font get base too much on filenames, please be smart. And missing list must not have any entry which is not in required list.
Only output valid JSON with key "missing".

uploaded: {json.dumps(uploaded_doc_types, ensure_ascii=False)}
required: {json.dumps(required_list, ensure_ascii=False)}

Return ONLY valid JSON:
{{"missing": ["..."]}}
"""
    llm_resp = get_LLM().invoke(prompt)
    llm_json = safe_json_loads(llm_resp, fallback={})
    missing_list = [
        x.strip() for x in llm_json.get("missing", [])
        if x and isinstance(x, str)
    ]
    return missing_list


def verify_checklist_dynamic(process_name, doc_types, kb_chunks=None, top_k=3):
    uploaded_doc_types = list(doc_types.values())
    required_list = fetch_required_docs_llm(process_name, kb_chunks=kb_chunks, top_k=top_k)
    missing_list = fetch_missing_docs_llm(uploaded_doc_types, required_list)
    return json.dumps({"required": required_list, "missing": missing_list}, ensure_ascii=False)
