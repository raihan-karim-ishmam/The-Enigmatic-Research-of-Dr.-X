# Phase 4 - Version 1.0 : RAG Q&A System - Retrieve and Answer (with flan-t5 models, switchable between small, base and large)

import os
import faiss
import json
import torch
import time
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM

# Set environment variable to avoid OpenMP error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Folder paths
DB_FOLDER = "vector_db"
CHUNKS_FOLDER = "chunks"

# Load FAISS index
index = faiss.read_index(os.path.join(DB_FOLDER, "chunks.index"))

# Load metadata
with open(os.path.join(DB_FOLDER, "metadata.json"), "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Load the Nomic embedding model
EMBED_MODEL = "nomic-ai/nomic-embed-text-v1"
tokenizer_embed = AutoTokenizer.from_pretrained(EMBED_MODEL, trust_remote_code=True)
model_embed = AutoModel.from_pretrained(EMBED_MODEL, trust_remote_code=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_embed = model_embed.to(device)

# ====== Answering LLMs - Choose one ======
# QA_MODEL = "google/flan-t5-small"
# QA_MODEL = "google/flan-t5-base"
QA_MODEL = "google/flan-t5-large"  # <-- ACTIVE currently

qa_tokenizer = AutoTokenizer.from_pretrained(QA_MODEL)
qa_model = AutoModelForSeq2SeqLM.from_pretrained(QA_MODEL).to(device)

# -------- FUNCTION TO EMBED A QUESTION -------- #

def embed_question(text):
    inputs = tokenizer_embed(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model_embed(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)
        return embedding.squeeze(0).cpu().numpy()

# -------- FUNCTION TO RETRIEVE CHUNKS -------- #

def retrieve_chunks(query_embedding, top_k=3):
    query_embedding = query_embedding.reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for idx in indices[0]:
        if idx < len(metadata):
            results.append(metadata[idx]["text"])
    return results

# -------- FUNCTION TO GENERATE AN ANSWER -------- #

def generate_answer(question, context_chunks):
    context = "\n\n".join(context_chunks)
    # Stronger prompt to force model to stick to context
    prompt = (
        f"Answer the question based ONLY on the context below.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        f"Answer:"
    )
    inputs = qa_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = qa_model.generate(**inputs, max_new_tokens=200)
    answer = qa_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# --------- MAIN LOOP ---------

print(f"✅ RAG Q&A System ready with {QA_MODEL}. Type your question or 'exit' to stop.")

while True:
    user_input = input("\nYour Question: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    start_time = time.time()

    # Embed question
    query_emb = embed_question(user_input)

    # Retrieve top-k chunks
    retrieved_chunks = retrieve_chunks(query_emb, top_k=3)

    # Generate answer
    final_answer = generate_answer(user_input, retrieved_chunks)

    end_time = time.time()
    total_time = end_time - start_time

    print(f"\n🧠 Answer: {final_answer}")
    print(f"⏱️ Time taken: {total_time:.2f} seconds")

    # Log performance
    log_entry = (
        f"==== Phase 4: Retrieval & Answering ({QA_MODEL}) ====\n"
        f"Question: {user_input}\n"
        f"Retrieved chunks: {len(retrieved_chunks)}\n"
        f"Total time taken: {total_time:.2f} seconds\n"
        "===========================\n\n"
    )

    with open("performance_log.txt", "a", encoding="utf-8") as log_file:
        log_file.write(log_entry)
