
"""ARRC Regulations Assistant (offline Streamlit app)

Quick start (once you've placed Regulations.pdf in ./data):
    streamlit run arrc_regulations_assistant.py

First run: click 'Build / Refresh Index' in the sidebar to create arrc_2025.faiss
"""

import os
import json
from pathlib import Path
import streamlit as st
import pdfplumber
import faiss
from sentence_transformers import SentenceTransformer, util

MODEL_NAME = "BAAI/bge-base-en-v1.5"
PDF_PATH = Path("Regulations.pdf")
INDEX_PATH = Path("arrc_2025.faiss")
META_PATH = Path("arrc_2025_meta.json")

CHUNK_SIZE = 120  # words
TOP_K = 3

@st.cache_resource(show_spinner=False)
def get_model():
    return SentenceTransformer(MODEL_NAME)

def pdf_to_chunks(pdf_path: Path, chunk_size=CHUNK_SIZE):
    if not pdf_path.exists():
        st.error(f"PDF not found at {pdf_path}")
        st.stop()
    text = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            text.append(page.extract_text() or "")
    full_text = "\n".join(text)
    words = full_text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i : i + chunk_size]
        chunk_text = " ".join(chunk_words)
        # Include first few words as 'title'
        title = " ".join(chunk_words[:8]) + "..."
        chunks.append({"text": chunk_text, "title": title})
    return chunks

def build_faiss_index(chunks, model, index_path=INDEX_PATH, meta_path=META_PATH):
    embeddings = model.encode([c["text"] for c in chunks], convert_to_numpy=True, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    faiss.write_index(index, str(index_path))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    return index

def load_faiss_index(index_path=INDEX_PATH):
    if not index_path.exists():
        return None
    return faiss.read_index(str(index_path))

def load_metadata(meta_path=META_PATH):
    if not meta_path.exists():
        return []
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)

def answer_query(query, model, index, metadata, top_k=TOP_K):
    if index is None:
        return "❗ Index not found. Build it first from the sidebar.", None
    q_emb = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)  # distances, indices
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        chunk = metadata[idx]
        results.append({"score": float(score), "text": chunk["text"]})
    if not results:
        return "Information not found in current regulations.", None
    # Simple answer: return the text of best chunk
    best = results[0]["text"]
    return best, results

# ---------- Streamlit UI ----------
st.set_page_config(page_title="ARRC Regulations Assistant", layout="wide")
st.title("🏁 ARRC Regulations Assistant")
model = get_model()

with st.sidebar:
    st.header("Index management")
    if st.button("Build / Refresh Index"):
        st.info("Building index... please wait ⏳")
        chunks = pdf_to_chunks(PDF_PATH)
        build_faiss_index(chunks, model)
        st.success("Index built ✅. You can now ask questions.")
    st.write("PDF path:", PDF_PATH)
    st.write("Index path:", INDEX_PATH)

index = load_faiss_index()
metadata = load_metadata()

query = st.text_input("Ask a question about the regulations:")
if query:
    answer, details = answer_query(query, model, index, metadata)
    st.write("### Answer")
    st.write(answer)
    if details:
        st.write("### Matching clauses")
        for r in details:
            st.write(f"- (score {r['score']:.2f}) {r['text'][:400]}…")
