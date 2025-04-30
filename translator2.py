# Phase 5 -Version 2.0 - Document Translator using NLLB-200 (to English/Arabic)
# Base inital version

import os
import time
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Define directories
INPUT_DIR = "input"
OUTPUT_DIR = "translated"
MODEL_REPO = "facebook/nllb-200-distilled-600M"
MODEL_DIR = "models/nllb-200-distilled-600M"
TARGET_LANG = "bn"  # Options: "en", "ar", "bn"

# Ensure folders exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Load or download model
print("🔄 Loading NLLB-200-distilled-600M model...")
if not os.path.exists(os.path.join(MODEL_DIR, "model.safetensors")):
    print("⬇️  Model not found locally. Downloading and saving to models folder...")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_REPO)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
else:
    print("✅ Model found locally. Loading from models folder.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

model.to("cpu")

# 🌐 Language tokens used by NLLB
LANG_CODES = {
    "en": "eng_Latn",
    "ar": "arb_Arab",
    "bn": "ben_Beng"
}

# Translation logic

def translate_text(text, lang):
    lang_code = LANG_CODES.get(lang, "eng_Latn")
    tokenizer.src_lang = "eng_Latn"
    inputs = tokenizer(text, return_tensors="pt", padding=True).to(model.device)
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(lang_code)
    start = time.time()
    generated_tokens = model.generate(**inputs, forced_bos_token_id=forced_bos_token_id)
    end = time.time()
    output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0].strip()
    return output, len(inputs.input_ids[0]), end - start

# Batch process files

def translate_all():
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".txt")]
    total_tokens = 0
    total_time = 0
    start_time = time.time()

    for file in files:
        input_path = os.path.join(INPUT_DIR, file)
        output_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(file)[0]}_translated.txt")

        with open(input_path, "r", encoding="utf-8") as f:
            source = f.read()

        translated, tokens_used, duration = translate_text(source, TARGET_LANG)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(translated)

        total_tokens += tokens_used
        total_time += duration
        print(f"✅ Translated: {file} → {output_path} ({tokens_used} tokens in {duration:.2f}s)")

    end_time = time.time()
    elapsed = end_time - start_time
    speed = total_tokens / total_time if total_time > 0 else 0

    # -------- SAVE PERFORMANCE LOG -------- #
    log_entry = (
        "==== Phase 5: Translation 2.0 ====" "\n"
        f"Model: {MODEL_REPO}\n"
        f"Target language: {TARGET_LANG}\n"
        f"Total documents translated: {len(files)}\n"
        f"Total tokens processed: {total_tokens}\n"
        f"Total LLM processing time: {total_time:.2f} seconds\n"
        f"Overall script runtime: {elapsed:.2f} seconds\n"
        f"Translation speed: {speed:.2f} tokens/second\n"
        "===========================\n\n"
    )
    with open("performance_log.txt", "a", encoding="utf-8") as log_file:
        log_file.write(log_entry)

# Print logs in the terminal for dev testing efficincy 

    print("\n==== Translation Performance ====")
    print(f"Documents translated: {len(files)}")
    print(f"Total tokens processed: {total_tokens}")
    print(f"Total LLM processing time: {total_time:.2f}s")
    print(f"Overall script runtime: {elapsed:.2f}s")
    print(f"Average speed: {speed:.2f} tokens/sec")
    print("================================\n")

if __name__ == "__main__":
    translate_all()
