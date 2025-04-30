# Phase 5 - Version 3.0 - Document Translator using Qwen2 Model (to English/Arabic)

import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

INPUT_DIR = "input"
OUTPUT_DIR = "translated"
MODEL_REPO = "Qwen/Qwen1.5-1.8B-Chat"
MODEL_DIR = "models/Qwen2-1.5B-Instruct"
TARGET_LANG = "bn"  # Change to "ar" or "en" for other targets

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"🔄 Loading {MODEL_REPO} model...")
try:
    if not os.path.exists(os.path.join(MODEL_DIR, "config.json")):
        print("⬇️  Model not found locally. Downloading...")
        model = AutoModelForCausalLM.from_pretrained(MODEL_REPO)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
        model.save_pretrained(MODEL_DIR)
        tokenizer.save_pretrained(MODEL_DIR)
    else:
        print("✅ Model found locally. Loading from models folder.")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype="auto", device_map="auto")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

def translate_text(text, lang):
    if lang == "en":
        prompt = f"Please translate the following text into English: {text}"
    elif lang == "ar":
        prompt = f"Please translate the following text into Arabic: {text}"
    elif lang == "bn":
        prompt = f"Please translate the following text into Bangla: {text}"
    else:
        prompt = text

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": prompt}
    ]
    templated = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([templated], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(inputs.input_ids, max_new_tokens=512)

    generated = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()

def translate_all():
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".txt")]
    total_tokens = 0
    start_time = time.time()

    for file in files:
        input_path = os.path.join(INPUT_DIR, file)
        output_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(file)[0]}_translated.txt")

        with open(input_path, "r", encoding="utf-8") as f:
            source = f.read()

        translated = translate_text(source, TARGET_LANG)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(translated)

        total_tokens += len(source.split())
        print(f"✅ Translated: {file} → {output_path}")

    elapsed = time.time() - start_time
    speed = total_tokens / elapsed if elapsed > 0 else 0

    log_entry = (
        "==== Phase 5: Translation 3.0 ====" "\n"
        f"Model: {MODEL_REPO}\n"
        f"Target language: {TARGET_LANG}\n"
        f"Total documents translated: {len(files)}\n"
        f"Total tokens processed: {total_tokens}\n"
        f"Total time taken: {elapsed:.2f} seconds\n"
        f"Translation speed: {speed:.2f} tokens/second\n"
        "===========================\n\n"
    )
    with open("performance_log.txt", "a", encoding="utf-8") as log_file:
        log_file.write(log_entry)

    print("\n==== Translation Performance ====")
    print(f"Documents translated: {len(files)}")
    print(f"Total tokens processed: {total_tokens}")
    print(f"Total time taken: {elapsed:.2f}s")
    print(f"Translation speed: {speed:.2f} tokens/sec")
    print("================================\n")

if __name__ == "__main__":
    translate_all()
