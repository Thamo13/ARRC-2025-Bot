# ARRC Regulations Assistant – Streamlit app (refined)
# --------------------------------------------------------------
# Key tweaks vs. first version
# • Smaller chunks (120‑word)  → tighter clause matches
# • Headline answer = first sentence of best chunk (plain, 1‑liner)
# • Still runs 100 % offline; no OpenAI key required
# • If you *do* add OPENAI_API_KEY to Streamlit Secrets, it will auto‑switch
#   to a GPT‑4o summary for an even cleaner 2‑line answer.

import os
import json
import re
from pathlib import Path
import textwrap
import streamlit as st
import pdfplumber
import faiss
from sentence_transformers import SentenceTransformer

# ---------------------------- config ----------------------------
MODEL_NAME = "BAAI/bge-base-en-v1.5"
PDF_PATH   = Path("Regulations.pdf")            # now looks in repo root
INDEX_PATH = Path("arrc_2025.faiss")
META_PATH  = Path("arrc_2025_meta.json")
CHUNK_SIZE = 120      # words per chunk (was 300)
TOP_K      = 3

# optional GPT polish
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    import openai
    openai.api_key = OPENAI_API_KEY

# ------------------------- helpers ------------------------------
@st.cache_resource(show_spinner=False)
def get_model():
    return SentenceTransformer(MODEL_NAME)

def pdf_to_chunks(pdf_path: Path, chunk_size: int = CHUNK_SIZE):
    pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    words = "\n".join(pages).split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        txt = " ".join(words[i : i + chunk_size])
        title = " ".join(words[i : i + 8]) + "…"
        chunks.append({"text": txt, "title": title})
    return chunks

def build_index(chunks, model):
    emb = model.encode([c["text"] for c in chunks], convert_to_numpy=True, show_progress_bar=True)
    faiss.normalize_L2(emb)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, str(INDEX_PATH))
    META_PATH.write_text(json.dumps(chunks, ensure_ascii=False))
    return index

def load_index():
    return faiss.read_index(str(INDEX_PATH)) if INDEX_PATH.exists() else None

def load_meta():
    return json.loads(META_PATH.read_text()) if META_PATH.exists() else []

# ---------------------- answer engine ---------------------------

def first_sentence(text: str) -> str:
    return re.split(r"[\.\n]", text, maxsplit=1)[0].strip()


def gpt_summary(context: str, question: str) -> str | None:
    if not OPENAI_API_KEY:
        return None
    resp = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Answer in 1‑2 short lines, plain English."},
            {"role": "user", "content": f"Regulation text:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return resp.choices[0].message.content.strip()


def answer_query(q: str, model, index, meta):
    if index is None:
        return "❗ Index not built yet.", []
    q_emb = model.encode([q], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, TOP_K)
    if I[0][0] == -1:
        return "Information not found in current regulations.", []
    best = meta[I[0][0]]["text"]
    headline = gpt_summary(best, q) or first_sentence(best)
    headline = textwrap.fill(headline, 90)
    return headline, [meta[idx]["text"] for idx in I[0] if idx != -1]

# --------------------------- UI --------------------------------
st.set_page_config(page_title="ARRC Regulations Assistant", layout="wide")
model = get_model()

with st.sidebar:
    st.header("Index management")
    if st.button("Build / Refresh Index"):
        if not PDF_PATH.exists():
            st.error(f"PDF not found at {PDF_PATH}")
            st.stop()
        st.info("Building index… please wait ⏳")
        idx = build_index(pdf_to_chunks(PDF_PATH), model)
        st.success("Index built ✅. You can now ask questions.")
    st.write("PDF path:", PDF_PATH.name)
    st.write("Index path:", INDEX_PATH.name)

index = load_index()
meta  = load_meta()

st.title("🏁 ARRC Regulations Assistant")
query = st.text_input("Ask a question about the regulations:")
if query:
    headline, clauses = answer_query(query, model, index, meta)
    st.subheader("Answer")
    st.write(headline)
    if clauses:
        st.subheader("Matching clause(s)")
        for txt in clauses:
            st.write("-", txt[:400] + ("…" if len(txt) > 400 else ""))
