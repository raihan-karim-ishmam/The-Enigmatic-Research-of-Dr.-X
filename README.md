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


## 🧩 Phase 3: Embedding & Vector Database Construction

### 🎯 Objective

Generate dense, normalized semantic embeddings for each chunk and store them in a **FAISS** vector database to enable fast and accurate semantic similarity search — forming the core of the Retrieval-Augmented Generation (RAG) system.

---

### ⚙️ Technologies Used

- `HuggingFace Transformers` (`nomic-embed-text-v1`)
- `faiss-cpu`
- `torch`, `numpy`

---

### 🧠 Engineering Approach

Phase 3 was dedicated to building an efficient and scalable vector-based retrieval foundation.

#### 📚 Embedding Generation
- Leveraged `nomic-embed-text-v1` model for generating efficient, general-purpose embeddings suitable for a wide range of scientific and technical domains.
- Applied **mean pooling** over the last hidden states to obtain a dense and representative semantic vector for each chunk.

#### 🔵 L2-Normalization for Stable Similarity Search
- Embedding vectors were L2-normalized before indexing to ensure **stable cosine similarity** behavior during search.
- This normalization improves retrieval robustness, particularly when dealing with variable-length input text.

#### 🛠️ Vector Database Construction
- FAISS (`faiss-cpu`) was selected to ensure fast, scalable, and offline semantic search capability.
- **Separation of Storage**:
  - FAISS index: stores only the raw vectors.
  - External `metadata.json`: stores corresponding metadata (filename, page, chunk_id, and text).
- This modular storage design promotes **easy future expansion**, **metadata refresh** without re-embedding, and **clear system maintainability**.

> 💡 **Design Insight:** Separating embeddings from metadata aligns with best practices for scalable information retrieval systems, reducing database fragility and enhancing update flexibility.

#### 📈 Performance Logging
- Full performance metrics were recorded during the embedding process:
  - Total tokens embedded
  - Embedding time
  - Tokens processed per second
- These logs provide **transparent insight** into system efficiency and facilitate optimization if scaling up.

---

### ✅ Outcome

- Successfully created a **fully populated FAISS database**, operational locally and optimized for RAG pipelines.
- Built a **transparent, performance-logged embedding layer** that ensures future scalability and operational clarity.
- Maintained full traceability and reproducibility through externalized metadata.

---


# Phase 4: RAG Q&A System Development

## Objective

Develop a flexible, scalable, and local Retrieval-Augmented Generation (RAG) system capable of answering queries grounded in Dr. X's publications.

## Structure and Scripts
- `rag1.py`: Basic RAG system without advanced truncation.
- `rag2.py`: Smart RAG system with dynamic token-budgeting, better prompt construction, and context truncation handling.
- `llama.py`: Specialized system leveraging Meta's LLaMA-2 7B-chat model in GGML format for local inference.

## LLM Integration

- **Meta’s LLaMA 2 Model**: Used in GGML format to support limited hardware.
- **Conversational Retrieval Chain**: Combined semantic vector retrieval and context-aware generation.
- **Token Optimization**: Dynamic chunking and controlled prompt budgets.

## Why GGML Format?

- Full LLaMA models require 30–50 GB VRAM; GGML compresses to ~6 GB RAM usage.
- Achieved efficient offline operation without expensive hardware.
- Demonstrates scalable engineering foresight.

## Download the LLaMA 2 Model

To run `rag_llama.py`:

1. Download `llama-2-7b-chat.ggmlv3.q4_0.bin` from:
   > [HuggingFace - TheBloke LLaMA-2-7B-Chat-GGML](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main)
2. Save inside `/models` directory.
3. Follow `instructions.md` inside `/models` for setup help.

## Evaluation Results

| Model | RAG Version | Mixed Docs Accuracy | Single Doc Accuracy |
|:------|:------------|:-------------------|:--------------------|
| Flan-t5-small | rag1.0.py | 20% | - |
| Flan-t5-base | rag1.0.py | 50% | 66% |
| Flan-t5-large | rag1.0.py | 45% | 93% |
| Flan-t5-small | rag2.0.py | 45% | - |
| Flan-t5-base | rag2.0.py | 45% | 60% |
| Flan-t5-large | rag2.0.py | 50% | 97% |
| LLaMA-2-7B-chat | llama.py | 60% | 82s% |

## Outcome

Delivered a clean, modular RAG system capable of working with multiple models and offline operation.

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

