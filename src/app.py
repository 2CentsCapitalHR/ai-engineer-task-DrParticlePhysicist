import streamlit as st
from pathlib import Path
import json

from doc_parser import extract_text_from_docx
from process_identifier import summarize_doc_with_kb, identify_process_from_summaries
from checklist_verification import verify_checklist_dynamic
from red_flag_detector import detect_red_flags_with_process
from inline_commenter import insert_comments
from output_generator import generate_summary_json
from Vectorstore_utils import search_Vectorstore  


def safe_json_loads(data, fallback=None):
    try:
        if isinstance(data, str):
            return json.loads(data)
        return data
    except Exception:
        return fallback if fallback is not None else data


st.title("📄 ADGM Compliance Review Agent")

# Step 0 - File upload
uploaded_files = st.file_uploader(
    "📤 Upload .docx files",
    accept_multiple_files=True,
    type=["docx"]
)

if uploaded_files:
    # ======================= STEP 0: File Reading =======================
    doc_texts = {}
    temp_paths = {}
    for uf in uploaded_files:
        temp_path = Path(f"temp_{uf.name}")
        with open(temp_path, "wb") as f:
            f.write(uf.getbuffer())
        temp_paths[uf.name] = temp_path
        doc_texts[uf.name] = extract_text_from_docx(temp_path)

    st.success(f"✅ {len(uploaded_files)} documents uploaded and parsed successfully.")

    # ======================= STEP 1: Summaries =======================
    summaries = {
        fname: summarize_doc_with_kb(text, fname)
        for fname, text in doc_texts.items()
    }

    # ======================= STEP 2: Process Identification =======================
    st.info("Identifying process & document types from summaries...")
    identification_raw = identify_process_from_summaries(summaries)
    identification = safe_json_loads(identification_raw, fallback={})

    st.subheader("🔍 Identified Process & Document Info")
    st.json(identification)

    process_name = identification.get("process", "")
    doc_types = identification.get("doc_types", {})
    process_doc_list = identification.get("process_doc_list", "")

    # ======================= STEP 3: Checklist Verification =======================
    if process_name:

        # Prepare KB query using process name + process doc list
        doc_str = ", ".join(process_doc_list) if isinstance(process_doc_list, list) else str(process_doc_list)
        kb_query = f"{process_name} {doc_str}".strip()

        # Get KB context
        kb_chunks_raw = search_Vectorstore(kb_query, top_k=5)
        kb_context_texts = [chunk["text"] for chunk in kb_chunks_raw if chunk.get("text")]


        checklist_raw = verify_checklist_dynamic(
            process_name,
            doc_types,
            kb_chunks=kb_context_texts
        )
        checklist_result = safe_json_loads(checklist_raw, fallback={"required": [], "missing": []})
    else:
        checklist_result = {"required": [], "missing": []}

    st.subheader("📋 Checklist Verification Result")
    st.json(checklist_result)


    # ======================= STEP 4: Red Flags Detection =======================
    st.info("Detecting red flags for each document...")
    reviewed_files, all_issues = [], []

    for fname, text in doc_texts.items():
        kb_for_doc_raw = search_Vectorstore(process_name + " " + text, top_k=5)
        kb_for_doc_texts = [chunk["text"] for chunk in kb_for_doc_raw if chunk.get("text")]

        issues_raw = detect_red_flags_with_process(process_name, text, fname, kb_for_doc_texts)
        issues = safe_json_loads(issues_raw, fallback=[])
        
        if not isinstance(issues, list):
            issues = []
        issues = [i for i in issues if isinstance(i, dict)]

        all_issues.extend([{"document": fname, **issue} for issue in issues])

        out_path = Path(f"reviewed_{fname}")
        insert_comments(temp_paths[fname], issues, out_path)
        reviewed_files.append(out_path)

    st.subheader("🚩 Red Flags Found")
    st.json(all_issues if all_issues else "No red flags detected.")

    # ======================= STEP 5: Summary JSON File =======================
    st.info("Generating compliance summary JSON...")
    summary_path = Path("compliance_summary.json")
    generate_summary_json(
        process_name,
        list(doc_texts.keys()),
        checklist_result.get("missing", []),
        checklist_result.get("required", []),
        all_issues,
        summary_path
    )
    st.success("Compliance summary generated.")

    # Show summary JSON in UI
    with open(summary_path, "r", encoding="utf-8") as f:
        summary_json = json.load(f)

    st.subheader("📜 Compliance Summary (JSON Report)")
    st.json(summary_json)

    # ======================= STEP 6: Download Reviewed DOCX only =======================
    st.subheader("⬇️ Download Reviewed Documents")
    for rf in reviewed_files:
        with open(rf, "rb") as f:
            st.download_button(f"📄 Download {rf.name}", f, file_name=rf.name)
