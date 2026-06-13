from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os
import tempfile
from typing import List

app = FastAPI(title="Research Paper Analyser API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global state ───────────────────────────────────────────────────────────────
embedder = SentenceTransformer("all-MiniLM-L6-v2")
faiss_index = None
chunks: List[str] = []
chat_history: List[dict] = []
llm_pipeline = None

# ─── Load LLM (Phi-3 or any local model) ───────────────────────────────────────
def load_llm():
    global llm_pipeline
    model_id = "microsoft/Phi-3-mini-4k-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.3,
        do_sample=True,
    )
    print("LLM loaded successfully.")

# Uncomment below to load LLM at startup (requires GPU/enough RAM)
# load_llm()

# ─── PDF Loading & Chunking ──────────────────────────────────────────────────────
def load_pdf(path: str) -> str:
    doc = fitz.open(path)
    return " ".join(page.get_text() for page in doc)

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    words = text.split()
    result = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        result.append(chunk)
        if i + chunk_size >= len(words):
            break
    return result

# ─── FAISS Indexing ──────────────────────────────────────────────────────────────
def build_faiss_index(text_chunks: List[str]):
    global faiss_index, chunks
    chunks = text_chunks
    embeddings = embedder.encode(text_chunks, show_progress_bar=True)
    dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index.add(np.array(embeddings, dtype="float32"))

def retrieve_chunks(query: str, top_k: int = 4) -> List[str]:
    if faiss_index is None:
        raise ValueError("No PDF indexed yet. Please upload a PDF first.")
    q_emb = embedder.encode([query]).astype("float32")
    _, indices = faiss_index.search(q_emb, top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]

# ─── LLM Answer Generation ───────────────────────────────────────────────────────
def generate_answer(question: str, context_chunks: List[str], history: List[dict]) -> str:
    context = "\n\n".join(context_chunks)
    history_text = ""
    for turn in history[-4:]:  # last 4 turns
        history_text += f"User: {turn['question']}\nAssistant: {turn['answer']}\n"

    prompt = f"""<|system|>
You are a research assistant. Answer questions strictly based on the provided context from a research paper.
If the answer is not in the context, say "This information is not available in the paper."
<|end|>
<|user|>
Context:
{context}

Previous conversation:
{history_text}

Question: {question}
<|end|>
<|assistant|>"""

    if llm_pipeline is None:
        # Fallback: return retrieved context summary if LLM not loaded
        return f"[LLM not loaded] Top relevant excerpt:\n\n{context_chunks[0] if context_chunks else 'No relevant content found.'}"

    result = llm_pipeline(prompt)
    generated = result[0]["generated_text"]
    answer = generated.split("<|assistant|>")[-1].strip()
    return answer

# ─── API Endpoints ────────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        text = load_pdf(tmp_path)
        if not text.strip():
            raise HTTPException(status_code=400, detail="PDF appears to be empty or unreadable.")
        text_chunks = chunk_text(text)
        build_faiss_index(text_chunks)
        chat_history.clear()
        return {
            "message": "PDF processed successfully.",
            "chunks": len(text_chunks),
            "filename": file.filename,
        }
    finally:
        os.unlink(tmp_path)


class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(req: QuestionRequest):
    if faiss_index is None:
        raise HTTPException(status_code=400, detail="Please upload a PDF before asking questions.")
    
    relevant = retrieve_chunks(req.question)
    answer = generate_answer(req.question, relevant, chat_history)
    
    chat_history.append({"question": req.question, "answer": answer})
    
    return {
        "question": req.question,
        "answer": answer,
        "sources": relevant[:2],  # Return top 2 source chunks for reference
    }

@app.delete("/reset")
async def reset_session():
    global faiss_index, chunks
    faiss_index = None
    chunks = []
    chat_history.clear()
    return {"message": "Session reset successfully."}

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "pdf_loaded": faiss_index is not None,
        "chunks_indexed": len(chunks),
    }