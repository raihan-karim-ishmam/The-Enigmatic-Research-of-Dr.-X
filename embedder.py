# Embedding and Storing Chunk Metadata in FAISS

import os
import json
import faiss
import torch
import numpy as np
import time
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm  # For progress bar

# Input and output folders
CHUNKS_FOLDER = "chunks"
DB_FOLDER = "vector_db"
os.makedirs(DB_FOLDER, exist_ok=True)

# Load the Nomic embedding model
EMBED_MODEL = "nomic-ai/nomic-embed-text-v1"

tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL, trust_remote_code=True)
model = AutoModel.from_pretrained(EMBED_MODEL, trust_remote_code=True)

# Move model to device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Initialize lists
all_embeddings = []
all_metadata = []

# -------- FUNCTION TO EMBED TEXT -------- #

def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
        return embedding.squeeze(0).cpu().numpy()

# -------- MAIN SCRIPT TO PROCESS CHUNKS -------- #

start_time = time.time()  # Start timing
total_tokens = 0  # Initialize token counter

# Go through all chunk files
for chunkfile in tqdm(os.listdir(CHUNKS_FOLDER), desc="Embedding chunks"):
    if not chunkfile.endswith(".json"):
        continue

    filepath = os.path.join(CHUNKS_FOLDER, chunkfile)
    with open(filepath, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    for chunk in chunks:
        text = chunk["text"]
        embedding = embed_text(text)

        all_embeddings.append(embedding)
        all_metadata.append({
            "filename": chunk["filename"],
            "page": chunk["page"],
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"]
        })

        total_tokens += len(tokenizer.encode(text))  # Count tokens

# Stack embeddings into a matrix
embedding_matrix = torch.tensor(all_embeddings).numpy()

# embedding_matrix = np.array(all_embeddings)   # No significant impact on our current set, optimal if chunks tally grows by far

# Create FAISS index
index = faiss.IndexFlatL2(embedding_matrix.shape[1])  # L2 distance (Euclidean)
index.add(embedding_matrix)

# Save FAISS index
faiss.write_index(index, os.path.join(DB_FOLDER, "chunks.index"))

# Save metadata separately
with open(os.path.join(DB_FOLDER, "metadata.json"), "w", encoding="utf-8") as f:
    json.dump(all_metadata, f, ensure_ascii=False, indent=2)

end_time = time.time()
total_time = end_time - start_time
tokens_per_second = total_tokens / total_time

print(f"✅ Finished embedding {len(all_metadata)} chunks and saving vector database!")
print(f"✅ Total tokens embedded: {total_tokens}")
print(f"✅ Total time taken: {total_time:.2f} seconds")
print(f"✅ Embedding speed: {tokens_per_second:.2f} tokens/second")

# -------- SAVE PERFORMANCE LOG -------- #

log_entry = (
    "==== Phase 3: Embedding ====\n"
    f"Total chunks embedded: {len(all_metadata)}\n"
    f"Total tokens embedded: {total_tokens}\n"
    f"Total time taken: {total_time:.2f} seconds\n"
    f"Embedding speed: {tokens_per_second:.2f} tokens/second\n"
    "===========================\n\n"
)

with open("performance_log.txt", "a", encoding="utf-8") as log_file:
    log_file.write(log_entry)
