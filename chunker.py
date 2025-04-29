# Phase 2 - Chunking text with [PAGE X] markers for proper page tracking

import os
import json
import tiktoken  # For tokenizing text like OpenAI models

# Folders where input and output files are located
INPUT_FOLDER = "output"
OUTPUT_FOLDER = "chunks"

# Create the output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Tokenizer setup
tokenizer = tiktoken.get_encoding("cl100k_base")  # Same tokenizer as GPT-4, GPT-3.5
MAX_TOKENS = 500  # Maximum tokens per chunk

# -------- HELPER FUNCTION TO CHUNK TEXT -------- #

def chunk_text(text, max_tokens):
    tokens = tokenizer.encode(text)
    chunks = []

    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)

    return chunks

# -------- MAIN SCRIPT TO PROCESS FILES -------- #

# Loop through all extracted .txt files
for filename in os.listdir(INPUT_FOLDER):
    if not filename.endswith(".txt"):
        continue

    file_path = os.path.join(INPUT_FOLDER, filename)

    try:
        print(f"Processing: {filename}")
        with open(file_path, "r", encoding="utf-8") as f:
            full_text = f.read()

        # Split text by [PAGE X] markers
        pages = full_text.split("[PAGE ")
        chunks_data = []
        chunk_id_counter = 0  # Global chunk counter per document

        for page_section in pages:
            if not page_section.strip():
                continue  # Skip empty parts

            # Get page number
            try:
                page_num_str, page_text = page_section.split("]", 1)
                page_num = int(page_num_str.strip())
            except ValueError:
                # If page marker is corrupted or missing, skip
                continue

            # Clean page text
            page_text = page_text.strip()

            # Chunk page text based on token limits
            chunks = chunk_text(page_text, MAX_TOKENS)

            for chunk in chunks:
                chunks_data.append({
                    "filename": filename,
                    "page": page_num,
                    "chunk_id": chunk_id_counter,
                    "text": chunk.strip()
                })
                chunk_id_counter += 1

        # Save output JSON
        out_name = filename.replace(".txt", ".json")
        output_path = os.path.join(OUTPUT_FOLDER, out_name)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)

        print(f"✅ Saved {len(chunks_data)} chunks for {filename}")

    except Exception as e:
        print(f"❌ Error with {filename}: {e}")
