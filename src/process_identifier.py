# src/process_identifier.py
import json
from Vectorstore_utils import search_Vectorstore
from LLM_client import get_LLM

def summarize_doc_with_kb(doc_text, filename, top_k=5):
    kb_chunks_raw = search_Vectorstore(doc_text, top_k=top_k)
    kb_context_texts = [c["text"] for c in kb_chunks_raw if c.get("text")]
    print(kb_context_texts)
    prompt = f"""
Summarize the following document for ADGM compliance review.

Document: {filename}

Relevant Knowledge Base context:
{json.dumps(kb_context_texts, indent=2, ensure_ascii=False)}

Document content excerpt:
{doc_text}

Focus on details and the part it is relevant to the Knowledge base.
List which tasks/process/procedure in ADGM paradigm the given docs are required for, 
and provide every important information to help process identification.

Provide a concise summary (max 200 words) and based on Knowledge base and do not include filename to use as information to make summary, as filename can be wrong and misleading.
"""
    return get_LLM().invoke(prompt)

def identify_process_from_summaries(summaries_dict):
    doc_blocks = "\n\n".join(f"{fname}: {summ}" for fname, summ in summaries_dict.items())

    prompt = f"""
You are an ADGM legal expert.

Below are multiple document summaries. 
For EACH summary/file, identify:
1. The overall process (if common for all uploaded docs)
2. The type of EACH document (doc_types must return a mapping for EVERY filename shown below)
3. A concise overall process summary focused on required document types.

Return ONLY valid JSON:
{{
  "process": "<process_name>",
  "doc_types": {{"filename1": "<doc_type>", "filename2": "<doc_type>", ...}},
  "process_summary": ""
}}

Summaries:
{doc_blocks}

IMPORTANT: The "doc_types" mapping MUST have an entry for EVERY summary/filename above, even if the type is "unknown".
"""
    return get_LLM().invoke(prompt)
