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

# Phase 1: Text Extraction & Preprocessing

## Objective

Extract structured, clean, page-tracked text from `.pdf`, `.docx`, `.csv`, and `.xlsx` files, ensuring readiness for downstream NLP tasks.

## Technologies Used
- PyPDF2 for PDF extraction
- python-docx for Word document extraction
- pandas, openpyxl, xlrd for CSV/Excel extraction

## Approach
- Modular extractors were developed per file type for flexibility.
- Page markers (`[PAGE x]`) were inserted during extraction.
- Dual output modes were created:
  - Token-optimized `.txt` files for processing.
  - Human-readable `.txt` files for manual inspection.
- An experimental `.json` export format was also explored.

## Outcome
- Successfully generated machine-friendly `.txt` files, preserving page structures.
- Clean baseline established for downstream chunking and semantic embedding.

---

# Phase 2: Chunking & Metadata Structuring

## Objective

Segment extracted text into manageable, semantically meaningful chunks while preserving origin metadata (file, page, chunk ID).

## Technologies Used
- tiktoken (`cl100k_base` tokenizer)
- JSON structured outputs

## Approach
- Token-based chunking to maintain semantic integrity (~500 tokens).
- Dynamic handling of oversized pages through truncation safety fallback.
- Full metadata recording: filename, page, chunk_id.

## Outcome
- Produced structured JSON outputs ready for efficient semantic embedding and retrieval.

---

# Phase 3: Embedding & Vector Database Construction

## Objective

Generate dense, normalized semantic embeddings for each chunk and store them in a FAISS vector database for fast similarity search.

## Technologies Used
- HuggingFace transformers (nomic-embed-text-v1)
- faiss-cpu
- torch, numpy

## Approach
- Mean pooling of hidden states.
- L2-normalization of embeddings.
- Separate storage of FAISS vector index and external metadata.
- Full performance logging (tokens/sec) for embedding phase.

## Outcome
- Fully populated, locally-operational FAISS database ready for Retrieval-Augmented Generation (RAG).

---

# Phase 4: RAG Q&A System Development

## Objective

Develop a flexible, scalable, and local Retrieval-Augmented Generation (RAG) system capable of answering queries grounded in Dr. X's publications.

## Structure and Scripts
- `rag1.0.py`: Basic RAG system without advanced truncation.
- `rag2.0.py`: Smart RAG system with dynamic token-budgeting, better prompt construction, and context truncation handling.
- `rag_llama.py`: Specialized system leveraging Meta's LLaMA-2 7B-chat model in GGML format for local inference.

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
| LLaMA-2-7B-chat | rag_llama.py | 60% | 93% |

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

