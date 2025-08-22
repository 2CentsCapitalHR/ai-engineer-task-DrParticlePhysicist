# 🏛️ ADGM Compliance Review Automation Agent

An AI-powered automation agent to streamline **compliance document review** for the Abu Dhabi Global Market (ADGM) regulatory framework.
Demo: https://drive.google.com/file/d/1Am9am0j_OmXm-_kGJ2_y6ABlRBHWSNvM/view?usp=drivesdk

---

## 🛠️ Setup

1. Clone repo  
   ```bash
   git clone <repo-url>
   cd adgm-compliance-agent
   ```

2. Create virtual environment  
   ```bash
   python -m venv .VirEnv
   .VirEnv\Scripts\activate   # Windows
   source .VirEnv/bin/activate  # Linux/Mac
   ```

3. Install dependencies  
   ```bash
   pip install -r requirements.txt
   ```

4. Configure `.env` in Project_root/src by setting 
HUGGINGFACEHUB_API_TOKEN=`your API token`(currently for the project, while u can change the models easily, plenty of room for customisation).

5. Run app  
   ```bash
   streamlit run src/app.py
   ```

---

## 🎯 Project Overview

This project provides an **end-to-end pipeline** to automatically:

- 📂 Upload corporate/legal documents (`.docx`)
- 🏷️ Identify process & document types
- 🔍 Semantic search with Knowledge Base
- ✅ Verify compliance checklists (ADGM specific)
- 🚨 Detect "red flag" compliance issues
- 📝 Insert inline comments inside `.docx`
- 📊 Generate JSON summaries & reviewed documents
- 🖥️ Use via a simple **Streamlit UI**

---

## ⚙️ Tech Highlights

- **Vector Search:** FAISS + SentenceTransformers
- **Chunking:** LangChain RecursiveCharacterTextSplitter
- **LLMs:** Integrated with HuggingFace/Perplexity API
- **Automation:** Inline comment insertion via `python-docx`
- **UI:** Streamlit/Gradio front-end
- **Reporting:** Downloadable `.docx` + JSON outputs

---

## 📂 Project Structure

```
src/
├── app.py                    # Main Streamlit/Gradio app
├── doc_parser.py              # Parse and extract .docx text
├── vectorstore_utils.py       # FAISS vectorstore load/query
├── process_identifier.py      # Classify process & document type
├── checklist_verification.py  # Verify docs vs ADGM checklist
├── red_flag_detector.py       # Flag compliance gaps
├── inline_commenter.py        # Insert inline comments in docx
├── output_generator.py        # Generate reviewed docx + JSON
```

---

## 💡 Usage

- Upload docs in UI  
- Review compliance output & red flags  
- Download annotated `.docx` + JSON report  

---

## 📜 License

MIT

---

🔥 *This project shows skills in NLP, vector search, and document automation for compliance workflows.*

---
