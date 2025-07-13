
"""ARRC Regulations Assistant – Streamlit app

Features
--------
· Works offline once the embedding model is cached (Sentence-Transformers bge-base-en-v1.5)
· Chunk size 120 words → focused retrieval
· Headline answer:
    - If OPENAI_API_KEY is set in env → GPT‑4o 2‑line summary
    - Else → first sentence of best matching chunk
· Always shows matching clause for citation
"""

from pathlib import Path
import os, re, textwrap, json
import streamlit as st
import pdfplumber
import faiss
from sentence_transformers import SentenceTransformer

# ---------------- Config ----------------
MODEL_NAME = "BAAI/bge-base-en-v1.5"
PDF_PATH   = Path("Regulations.pdf")        # PDF in repo root
INDEX_PATH = Path("arrc_2025.faiss")
META_PATH  = Path("arrc_2025_meta.json")
CHUNK_SIZE = 120                            # words
TOP_K      = 3

# --------------- Helpers ----------------
@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer(MODEL_NAME)

def pdf_to_chunks(pdf_path: Path, chunk_size: int):
    pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    words = " ".join(pages).split()
    chunks, meta = [], []
    for i in range(0, len(words), chunk_size):
        text = " ".join(words[i:i + chunk_size])
        chunks.append(text)
        meta.append({"text": text})
    return chunks, meta

def build_index(text_chunks, meta):
    model = load_model()
    emb = model.encode(text_chunks, convert_to_numpy=True, show_progress_bar=True)
    faiss.normalize_L2(emb)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, str(INDEX_PATH))
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False))
    return index

def load_index():
    return faiss.read_index(str(INDEX_PATH)) if INDEX_PATH.exists() else None

def load_meta():
    return json.loads(META_PATH.read_text()) if META_PATH.exists() else []

# ---------- Answer logic ----------
def first_sentence(text: str) -> str:
    sent = re.split(r"[\.\\n]", text, maxsplit=1)[0].strip()
    # Strip common lead-ins
    sent = re.sub(r"^(as below|as per below|see below)\\s*\\d*[:\\-]?\\s*", "", sent, flags=re.I)
    return sent


def gpt_summary(context: str, question: str):
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    import openai
    openai.api_key = key
    resp = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Answer in 2 short lines, plain English."},
            {"role": "user", "content": f"Regulation text:\n{context}\n\nQuestion: {question}"}
        ],
        temperature=0
    )
    return resp.choices[0].message.content.strip()

def answer_query(q: str, index, meta):
    if index is None:
        return "❗ Index not built yet.", None
    model = load_model()
    q_emb = model.encode([q], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, TOP_K)
    if I[0][0] == -1:
        return "Information not found in current regulations.", None
    best = meta[I[0][0]]["text"]
    headline = gpt_summary(best, q) or first_sentence(best)
    return headline, best

# -------------- Streamlit UI --------------
st.set_page_config(page_title="ARRC Regulations Assistant", layout="wide")
st.title("🏁 ARRC Regulations Assistant")

with st.sidebar:
    st.header("Index management")
    if st.button("Build / Refresh Index"):
        if not PDF_PATH.exists():
            st.error(f"PDF not found at {PDF_PATH}")
            st.stop()
        st.info("Building index… this may take up to a minute.")
        chunks, meta = pdf_to_chunks(PDF_PATH, CHUNK_SIZE)
        build_index(chunks, meta)
        st.success("Index built ✅. You can now ask questions.")
    st.markdown(f"**PDF path:** `{PDF_PATH.name}`  \\n**Index path:** `{INDEX_PATH.name}`")

index = load_index()
meta = load_meta()

query = st.text_input("Ask a question about the regulations:")
if query:
    answer, clause = answer_query(query, index, meta)
    st.markdown("### Answer")
    st.write(answer)
    if clause:
        st.markdown("---\n### Matching clause")
        st.write(clause)
