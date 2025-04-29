# ======================================================================================
# 🟩 Batch Document Summarizer + ROUGE Evaluation + Performance Logging (PDF, DOCX, TXT)
# --------------------------------------------------------------------------------------
# This script summarizes all .pdf, .docx, and .txt files inside the ./input/ folder.
# It saves summaries in ./summaries/, ROUGE metrics in ./rouge_metrics/, and performance in performance_log.txt.
# Model used: LaMini-Flan-T5-248M (from ./models/ folder)
# ======================================================================================

import os
import time
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import torch
from evaluate import load

# 📂 Define input, output, and model directories
INPUT_DIR = "input"
SUMMARY_DIR = "summaries"
ROUGE_DIR = "rouge_metrics"
MODEL_DIR = "models/LaMini-Flan-T5-248M"

# 🧱 Create folders if they don't exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)
os.makedirs(ROUGE_DIR, exist_ok=True)

# 🤖 Load model and tokenizer
tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
base_model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR, device_map='auto', torch_dtype=torch.float32)
rouge = load("rouge")

# 🛠 File preprocessing: load and split text depending on file type
def file_preprocessing(file_path):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    final_texts = ""
    for text in texts:
        final_texts += text.page_content
    return final_texts

# 🧠 LLM summarization pipeline
def llm_pipeline(text):
    summarizer = pipeline(
        'summarization',
        model=base_model,
        tokenizer=tokenizer,
        max_length=500,
        min_length=50
    )
    result = summarizer(text)
    return result[0]['summary_text']

# 🏗 Main batch processing function
def summarize_all_documents():
    document_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(('.pdf', '.docx', '.txt'))]

    total_tokens = 0
    start_time = time.time()

    for file in tqdm(document_files, desc="Summarizing Documents", ncols=100):
        input_path = os.path.join(INPUT_DIR, file)
        base_name = os.path.splitext(file)[0]
        output_path = os.path.join(SUMMARY_DIR, f"{base_name}_summary.txt")
        rouge_path = os.path.join(ROUGE_DIR, f"{base_name}_rouge.txt")

        full_text = file_preprocessing(input_path)
        summary = llm_pipeline(full_text)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(summary)

        scores = rouge.compute(predictions=[summary], references=[full_text])
        with open(rouge_path, "w", encoding="utf-8") as f:
            for metric, value in scores.items():
                line = f"{metric}: {value:.4f}\n"
                f.write(line)

        total_tokens += len(full_text.split())

    end_time = time.time()
    total_time = end_time - start_time
    tokens_per_second = total_tokens / total_time if total_time > 0 else 0

    # -------- SAVE PERFORMANCE LOG -------- #
    log_entry = (
        "==== Phase 6: Summarization ====" "\n"
        f"Total documents summarized: {len(document_files)}\n"
        f"Total tokens processed: {total_tokens}\n"
        f"Total time taken: {total_time:.2f} seconds\n"
        f"Summarization speed: {tokens_per_second:.2f} tokens/second\n"
        "===============================\n\n"
    )

    with open("performance_log.txt", "a", encoding="utf-8") as log_file:
        log_file.write(log_entry)

if __name__ == "__main__":
    summarize_all_documents()
