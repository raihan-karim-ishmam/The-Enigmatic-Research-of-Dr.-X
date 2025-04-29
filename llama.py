# Phase 4: RAG Q&A System - Using Local Quantized LLaMA 2 7B Chat Model

import os
import faiss
import json
import time
import torch
from ctransformers import AutoModelForCausalLM
import numpy as np

# Set environment variable to avoid OpenMP error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Folder paths
DB_FOLDER = "vector_db"
CHUNKS_FOLDER = "chunks"

# Path to your local quantized llama model
LLAMA_MODEL_PATH = "models/llama-2-7b-chat.ggmlv3.q4_0.bin"


# Load FAISS index
index = faiss.read_index(os.path.join(DB_FOLDER, "chunks.index"))

# Load metadata
with open(os.path.join(DB_FOLDER, "metadata.json"), "r", encoding="utf-8") as f:
    metadata = json.load(f)

# -------- Load Local Quantized LLaMA 7B Model -------- #

llm = AutoModelForCausalLM.from_pretrained(
    LLAMA_MODEL_PATH,
    model_type="llama",
    gpu_layers=0,  # All CPU inference
    context_length=2048,
)

# -------- History setup -------- #
history = []
MAX_HISTORY = 3  # Remember last 3 exchanges

# -------- FUNCTION TO EMBED A QUESTION (Simple Average Word Vector) -------- #
# (Assumes embeddings already handled during FAISS building.)

def embed_question(text):
    # For simplicity, use FAISS vector space directly (you already have embedding model for this)
    from transformers import AutoTokenizer, AutoModel
    EMBED_MODEL = "nomic-ai/nomic-embed-text-v1"
    tokenizer_embed = AutoTokenizer.from_pretrained(EMBED_MODEL, trust_remote_code=True)
    model_embed = AutoModel.from_pretrained(EMBED_MODEL, trust_remote_code=True)
    device = "cpu"
    model_embed = model_embed.to(device)

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

def generate_answer(question, context_chunks, history):
    context = "\n\n".join(context_chunks)

    # Build memory
    memory_text = ""
    for past_qa in history[-MAX_HISTORY:]:
        memory_text += f"Previous Question: {past_qa['question']}\nPrevious Answer: {past_qa['answer']}\n\n"

    prompt = (
        f"{memory_text}"
        f"Use ONLY the following context to answer the new question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        f"Answer:"
    )

    output = llm(prompt, max_new_tokens=200)
    return output

# --------- MAIN LOOP ---------

print("✅ RAG Q&A System ready with LLaMA-2-7B-Chat (Quantized)! Type your question or 'exit' to stop.")

while True:
    user_input = input("\nYour Question: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    start_time = time.time()

    # Embed question
    query_emb = embed_question(user_input)

    # Retrieve top-k chunks
    retrieved_chunks = retrieve_chunks(query_emb, top_k=3)

    # Generate answer with memory
    final_answer = generate_answer(user_input, retrieved_chunks, history)

    end_time = time.time()
    total_time = end_time - start_time

    print(f"\n🧠 Answer: {final_answer}")
    print(f"⏱️ Time taken: {total_time:.2f} seconds")

    # Save into retrieval log
    retrieval_entry = (
        "==== Retrieval Log (LLaMA 7B) ====\n"
        f"Question: {user_input}\n"
        f"Retrieved Chunks:\n"
    )
    for idx, chunk in enumerate(retrieved_chunks, 1):
        retrieval_entry += f"\nChunk {idx}:\n{chunk}\n"
    retrieval_entry += f"\nGenerated Answer:\n{final_answer}\n"
    retrieval_entry += f"Time Taken: {total_time:.2f} seconds\n"
    retrieval_entry += "========================\n\n"

    with open("retrieval_log.txt", "a", encoding="utf-8") as log_file:
        log_file.write(retrieval_entry)

    # Save performance log
    log_entry = (
        "==== Phase 4: Retrieval & Answering (LLaMA 7B) ====\n"
        f"Question: {user_input}\n"
        f"Retrieved Chunks: {len(retrieved_chunks)}\n"
        f"Total Time Taken: {total_time:.2f} seconds\n"
        "===========================\n\n"
    )

    with open("performance_log.txt", "a", encoding="utf-8") as perf_file:
        perf_file.write(log_entry)

    # Update chat history
    history.append({
        "question": user_input,
        "answer": final_answer
    })
