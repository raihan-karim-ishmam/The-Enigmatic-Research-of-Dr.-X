# 📃 Instructions: Downloading Required Models for OSOS NLP System

## Overview
This file provides step-by-step instructions for downloading and placing the required local models in the `models` directory. Due to their large size, these models are **not included in the repository** and must be manually downloaded.

---

## Requirements
Before you begin, ensure you have the following:

- **Python** (Version 3.8 or above)
- **At least 15 GB of free disk space**
- **Stable internet connection**
- **(Optional)** Git LFS installed if cloning model repositories

---

## 📄 Downloading the LLaMA 2 Model (Phase 4)

To use the system, you need to download the **LLaMA 2 Model** in GGML format. Follow these steps:

1. Navigate to the following link:  
   [Llama 2 Model on HuggingFace](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main)

2. Download the model file:
   ```
   llama-2-7b-chat.ggmlv3.q4_0.bin
   ```

3. Save the downloaded file to the `models` directory in the repository.

4. Ensure the model file is located at:
   ```
   <repository-root>/models/llama-2-7b-chat.ggmlv3.q4_0.bin
   ```

---

## 🔗 Downloading the LaMini-Flan-T5 Model (Phase 6)

To perform summarization and other LLM tasks, the project uses the **LaMini-Flan-T5-248M** model from MBZUAI. Follow these steps:

1. Visit the model link:  
   [LaMini-Flan-T5-248M on HuggingFace](https://huggingface.co/MBZUAI/LaMini-Flan-T5-248M)

2. You have two options to download:
   - **Option A** (Recommended): Use `git clone` to download the full repository:
     ```bash
     git clone https://huggingface.co/MBZUAI/LaMini-Flan-T5-248M
     ```
   - **Option B**: Manually download all files from the model page.

3. Place the contents inside a folder with the following path:
   ```
   <repository-root>/models/LaMini-Flan-T5-248M/
   ```

> 🔄 **Note**: The folder name **must** be `LaMini-Flan-T5-248M` for the application to locate the model correctly.

---

## 🔹 (Optional) Downloading the TinyLLaMA Model

This model was used in the **experimentation and building phase**, but is not required to run the current pipeline. If you would like to include it, follow these steps:

1. Navigate to the model page:  
   [TinyLlama-1.1B-Chat-v0.6-GGUF on HuggingFace](https://huggingface.co/afrideva/TinyLlama-1.1B-Chat-v0.6-GGUF)

2. Download the following file:
   ```
   tinyllama-1.1b-chat-v0.6.Q4_K_M.gguf
   ```

3. Save the file inside the `models` directory:
   ```
   <repository-root>/models/tinyllama-1.1b-chat-v0.6.Q4_K_M.gguf
   ```

> 💡 **Optional Use**: This model is not actively used in the core scripts and is only relevant for testing or optional lightweight experimentation.

---

## Notes

- **Do not rename model files or folders.** The code relies on specific paths and names.
- **Ensure correct placement** of all models before running the system.
- **Minimum Requirements**: LLaMA 2 GGML models require 16GB RAM for smooth operation on CPU.

---

## 🌐 Future Improvements

- Switch to quantized GPU versions for better runtime performance.
- Automate model download and validation via script with HuggingFace authentication.
- Introduce model management UI in the app for selection and swapping.

---

For further details, refer to the main `README.md` in the repository.

