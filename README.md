# The Enigmatic Research of Dr. X
*AI-Powered NLP Pipeline for OSOS*

![Python](https://img.shields.io/badge/Python-3.10-blue) ![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow) ![FAISS](https://img.shields.io/badge/Vector-Database-FAISS-blue) ![Local-NLP](https://img.shields.io/badge/Deployment-Local_Only-green) ![NLP](https://img.shields.io/badge/Field-NLP-red)

---

# Project Overview

The **Enigmatic Research of Dr. X** is a detailed, multi-phase AI engineering project completed for **OSOS**, a pioneering AI company based in Oman. Centered around the fictional mystery of Dr. X's disappearance, the challenge was to create a **robust, fully offline, NLP-driven Q&A system** capable of reading, understanding, and interacting with Dr. X's research publications.

The system was built under strict local-only hardware constraints, emphasizing scalability, creativity, and real-world production quality throughout every phase.

---

# Bonus Extension: Structured Data (Excel/CSV) Support

While the original assignment only required handling `.docx` and `.pdf` documents, this solution was extended to fully support **structured Excel and CSV files** (`.csv`, `.xlsx`, `.xls`, `.xlsm`).

- Full pipeline support across extraction, chunking, embedding, and retrieval.
- Significantly boosts real-world scalability by including tabular sources.
- Some minor QA accuracy drops were noted from tabular noise, but overall system robustness improved.

**This extension is a major bonus that showcases foresight and real-world engineering adaptability.**

---

# Installation Guide

To set up and run the project locally:

```bash
# 1. Clone the repository
$ git clone https://github.com/yourusername/enigmatic-research-dr-x.git

# 2. Navigate into the project directory
$ cd enigmatic-research-dr-x

# 3. Install required dependencies
$ pip install -r requirements.txt

# 4. (Optional) Download the LLaMA-2 GGML model manually if using rag_llama.py
```

Detailed instructions for LLaMA model setup are provided below.

---

## 🧩 Phase 1: Text Extraction & Preprocessing

### 🎯 Objective

Extract structured, page-tracked, and cleanly formatted text from `.pdf`, `.docx`, `.csv`, and `.xlsx`/`.xlsm` files — ensuring the extracted content is immediately usable for downstream NLP tasks like tokenization, embedding, and retrieval.

---

### ⚙️ Technologies Used

- `PyPDF2` for PDF extraction  
- `python-docx` for DOCX parsing  
- `pandas`, `openpyxl`, `xlrd` for handling Excel/CSV data  

Each extractor was isolated into modular functions for flexibility and future extension.

---

### 🧠 Engineering Approach

This phase was designed to build a **reliable and extensible foundation**. The core design philosophy revolved around **preserving semantic meaning** while maintaining a **uniform structure** across diverse document formats.

#### 📁 File-Type Specific Extractors
- A dispatch function determines the correct extraction method based on file extension.
- Extraction logic is customized per file type to ensure optimal parsing — especially handling multi-sheet Excel files and page simulation in Word/Excel.

#### 📌 Page Marking for Traceability
- Inserted `[PAGE X]` markers to simulate or preserve pagination — crucial for:
  - Accurate context referencing during chunking
  - Traceability during answer generation
  - Fine-grained summarization control

#### 🔄 Output Versions
To serve different downstream needs, **three parallel output modes** were engineered:

| Version | Format | Purpose |
|--------|--------|---------|
| **1.0** | `.txt` (token-optimized) | Dense format, no extra whitespace – optimized for chunking, embedding |
| **1.1** | `.txt` (aligned) | Readable, column-aligned format – useful for inspection/debugging |
| **1.2** | `.json` | Structured format for programmatic analysis or future UI integration |

> 💡 **Design Insight:** This multi-version approach shows thoughtful separation of concerns — optimizing for **both machine processing** and **human debugging** without sacrificing either.

#### 🧪 Simulated Pagination Logic
- `.docx`: simulated 1 page every 50 paragraphs  
- `.csv`/`.xlsx`: simulated 1 page per 30 rows  
- `.pdf`: true page tracking from document metadata

---

### ✅ Outcome

- Achieved **consistent, structured output** for all supported file types.
- Created modular and extensible extractors that can scale as the project grows.
- Established a **clean, traceable baseline** for Phase 2 chunking and semantic embedding.
- Optimized for both **human legibility** and **NLP-readiness**.


---


## 🧩 Phase 2: Chunking & Metadata Structuring

### 🎯 Objective

Segment the extracted text into manageable, semantically meaningful chunks while preserving origin metadata, including filename, page number, and chunk ID — preparing the dataset for efficient embedding and retrieval workflows.

---

### ⚙️ Technologies Used

- `tiktoken` (`cl100k_base` tokenizer)  
- Structured `JSON` outputs

---

### 🧠 Engineering Approach

The chunking strategy was designed with a strong focus on **semantic integrity** while respecting **token constraints** required for downstream LLM usage.

#### 📚 Token-Based Chunking
- Text was tokenized using `cl100k_base`, ensuring compatibility with modern models like GPT-4 and GPT-3.5.
- Targeted chunk size: approximately **500 tokens** per chunk — balancing context richness with performance.

#### 🛡️ Safety Fallbacks
- Dynamic handling of oversized pages:
  - If a single page's text exceeded the maximum token limit, it was **safely split** into multiple chunks without losing important context boundaries.

#### 📋 Full Metadata Structuring
Every chunk carries the following metadata:
- `filename`: Source file of the chunk
- `page`: Page number from where the chunk originated
- `chunk_id`: Sequential ID for traceability
- `text`: Chunk content (cleaned and token-bounded)

This metadata-driven design ensures **traceability**, **easy reassembly**, and **fine-grained retrieval** in future stages.

> 💡 **Design Insight:** By embedding detailed metadata early at the chunking stage, the pipeline remains modular and scalable for advanced features like search relevance scoring, result backtracking, and explainability.

---

### ✅ Outcome

- Successfully produced **structured and metadata-rich JSON outputs**.
- Generated chunks are **optimized for semantic embedding**, **retrieval**, and **question answering tasks**.
- Established a **tokenization-consistent** foundation critical for later phases (embedding generation and RAG system construction).


---


# Phase 3: Embedding & Vector Database Construction

## Objective

Generate dense, normalized semantic embeddings for each chunk and store them in a FAISS vector database for fast similarity search — forming the foundation for the Retrieval-Augmented Generation (RAG) system.

## Technologies Used

- `HuggingFace Transformers` (`nomic-embed-text-v1`)
- `faiss-cpu`
- `torch`
- `numpy`

## Engineering Approach

### 📚 Embedding Generation

- Used `nomic-embed-text-v1`, a strong general-purpose embedding model, suitable for scientific and multi-domain documents.
- Applied mean pooling across the final hidden states to generate dense semantic vectors for each chunk.

### 🔵 L2-Normalization for Similarity Search

- Embedding vectors were L2-normalized prior to indexing.
- This ensured stable cosine similarity behavior during retrieval, improving robustness across variable-length inputs.

### 🛠️ Vector Database Construction

- Built the vector database using `faiss-cpu`, selected for its speed, reliability, and full offline capability.
- Storage strategy was deliberately separated:
  - **FAISS Index**: Stores dense vector representations.
  - **External Metadata File**: Stores associated metadata (filename, page, chunk_id, text) in a structured `metadata.json`.

> **💡 Design Insight:** This separation between vectors and metadata allows for scalable system upgrades without needing full re-indexing when only metadata changes.

### 📈 Performance Logging

- Full performance metrics were logged during the embedding phase:
  - Total tokens embedded
  - Time taken
  - Embedding speed (tokens processed per second)

This provides transparency on system efficiency and allows easy benchmarking for future scaling.

## Outcome

- Successfully created a fully populated, locally-operational FAISS vector database.
- Built a modular, transparent, and performance-logged embedding layer.
- Ensured future scalability, allowing for easy expansions, metadata refreshes, and optimizations.


---


# Phase 4: RAG Q&A System Development

## Objective

Develop a flexible, scalable, and local Retrieval-Augmented Generation (RAG) system capable of answering queries grounded in Dr. X's publications, fully operational on minimal hardware resources without reliance on external APIs.

## Technologies Used

- `HuggingFace Transformers`
- `faiss-cpu`
- `torch`
- `ctransformers` (for running local LLaMA-2 models)
- `nomic-embed-text-v1` (embedding model)
- `flan-t5-small`, `flan-t5-base`, `flan-t5-large` (switchable)
- `LLaMA-2-7B-Chat` (quantized GGML format)

## Structure and Scripts

- `rag.py`: Base RAG system with Flan-T5 answering and static prompt context.
- `rag2.py`: Enhanced system with dynamic token-budgeting, tokenizer-safe truncation, and better prompt engineering.
- `llama.py`: Offline-only RAG system using the locally quantized Meta `LLaMA-2-7B-Chat` model (GGML) via `ctransformers`.

---

## LLM Integration and Model Choices

### 📚 Model Choices and Reasoning

- **Flan-T5 Series** (`small`, `base`, `large`): Used for early development and testing due to their instruction-following capability and ability to run on CPUs with minimal memory (especially `flan-t5-small`).
- **Meta LLaMA-2-7B-Chat (GGML)**: Chosen for full local operation, with no external dependencies. The model was downloaded in `ggmlv3.q4_0.bin` quantized format to allow CPU-based inference under 16GB RAM.

> **💡 Installation Note:**  
> To use LLaMA locally, manually download `llama-2-7b-chat.ggmlv3.q4_0.bin` and place it in a `models/` directory. The system does not auto-download models to preserve disk space and maintain a lightweight, offline setup.

---

## Engineering Approach

### 🛠️ Retrieval Strategy Enhancements

- **Dynamic Top-K Retrieval**: Retrieves top 3 chunks relevant to the question using FAISS cosine similarity.
- **Normalized Question Embedding**: Improves search consistency across question types and lengths.

### 📚 Prompt Construction and Context Handling

- **Structured Prompting**: Clear instructions like `"Answer the question based ONLY on the context below"` were used to reduce hallucinations.
- **Dynamic Context Building (rag2.py)**: Chunks are added progressively until the token budget (~512–800 tokens) is reached. Prevents overloading the model with unnecessary context.
- **Tokenizer-Level Truncation**: Ensures that truncation preserves semantic integrity by using the tokenizer, avoiding raw string slicing.

### 🔒 Local Inference and Environment Controls

- **Isolated Per-Question Execution**: No memory between questions. Each query starts with a clean context.
- **Error-Safe Execution**: Known CPU issues (e.g., OpenMP duplication) are handled with environment flags.
- **Model Switching**: With a one-line edit, developers can switch between `small`, `base`, `large`, or `llama` models depending on task and resources.

### 📈 Performance Monitoring

- Logs total retrieval and answering time per question.
- Retrieval logs and answer logs are saved independently for reproducibility.

---

## Outcome

- Fully offline, local RAG Q&A system successfully built and tested.
- Demonstrated that meaningful document-grounded question answering can be achieved without any external APIs or cloud-based inference.
- Maintained end-to-end traceability of retrieved context and generated answers.
- Final average CPU response time:  
  - ~15–30 seconds for Flan models  
  - ~30+ seconds for LLaMA 7B quantized GGML model

---

## Limitations and Hardware Constraints

- **Local Development Environment**:
  - No GPU usage (GTX 1050 Ti available but unused)
  - 16GB RAM and limited storage
- **Resulting Trade-Offs**:
  - Only quantized models could fit in memory (no 13B or 65B LLaMA support)
  - Inference time slower than GPU setups
  - Flan-T5 performed reasonably well, but lacks deep reasoning found in large models like GPT-4 or Claude

> **🔒 Design Tradeoff:**  
> Offline inference was prioritized as per the assessment requirements — despite potential for better accuracy via APIs like OpenAI or Anthropic, these were deliberately excluded.

---

## Future Improvements and Production Scaling Ideas

- **🚀 Hardware Upgrade**:
  - Upgrading to a modern GPU (e.g., RTX 4090) and 64GB+ RAM would allow:
    - Running full precision LLaMA models (7B+)
    - Much faster answering (2–3 sec range)
    - Fine-tuning capabilities

- **🧠 Fine-Tuning**:
  - Train Flan or LLaMA on Dr. X’s publications to create domain-adapted instruction-following models.
  - Customize prompt formats for different publication types.

- **🔎 Enhanced Retrieval**:
  - Combine semantic retrieval with keyword matching (hybrid search)
  - Introduce query rewriting for better chunk matching

---


# Engineering Enhancements and Optimization Choices

## Phase 1 Highlights
- Modular format-specific extraction.
- Page markers inserted during reading.
- Tabular-to-text transformation for CSV/Excel.

## Phase 2 Highlights
- Token-based chunking.
- Full metadata tagging.
- Dynamic fallback for long pages.

## Phase 3 Highlights
- Smart embedding normalization.
- Vector/metadata separation for scalability.
- Performance metrics logged.

## Phase 4 Highlights
- Dynamic prompt token control.
- Smart Top-K chunk retrieval.
- Model-agnostic RAG script structure.
- Realistic simulation and LLaMA extension.

---

*Case developed by Raihan Karim for OSOS. © 2025 Raihan Karim. All rights reserved.*

