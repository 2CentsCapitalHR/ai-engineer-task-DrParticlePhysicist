# src/Vectorstore_utils.py
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstores" / "faiss_index"
INDEX_FILE = VECTORSTORE_DIR / "faiss_index.bin"
META_FILE = VECTORSTORE_DIR / "metadata.pkl"

def load_Vectorstore():
    if not INDEX_FILE.exists() or not META_FILE.exists():
        raise FileNotFoundError("Index or metadata file not found.")
    index = faiss.read_index(str(INDEX_FILE))
    with open(META_FILE, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

def search_Vectorstore(query_text, top_k=5):
    index, metadata = load_Vectorstore()
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    query_emb = np.array(model.encode([query_text])).astype("float32")

    distances, indices = index.search(query_emb, top_k)
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx == -1:
            continue
        meta = metadata[idx]
        results.append({
            "text": meta.get("text", "[No chunk text stored]"),
            "metadata": meta,
            "score": float(dist)
        })
    return results
