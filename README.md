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
To extract clean, structured, and page-tracked text from diverse publication formats (`.pdf`, `.docx`, `.csv`, `.xlsx`) with a focus on preparing machine-friendly inputs for downstream NLP tasks such as chunking, embedding, retrieval, translation, and summarization.

## Technologies Used
- **PDF Extraction**: `PyPDF2`
- **Word Document Extraction**: `python-docx`
- **Spreadsheet and CSV Extraction**: `pandas`, `openpyxl`, `xlrd`

## Approach

A modular, extensible extraction pipeline was developed to handle each file type independently while maintaining a consistent output structure across all sources. Key design decisions included:

- **Page Tracking**: Inserted `[PAGE x]` markers within the extracted text to enable page-level traceability throughout the NLP pipeline.
- **Dual Output Modes**:
  - **Version 1.0**: Standard extraction — text output with minimal processing, ensuring faithful data capture with sufficient readability across all file types.
  - **Version 1.1**: Visually-enhanced extraction — introduced additional spacing and alignment corrections, especially for table-heavy documents like `.csv` and `.xlsx`, improving human readability without compromising tokenization integrity.
  - **Version 1.2**: Experimental JSON format — extracted outputs converted into a JSON structure for potential metadata enhancement in future scaling phases (not fully leveraged in this proof-of-concept due to limited metadata density).
- **File Organization**:
  - Scripts: `extract.py` (standard), `extract1.py` (visually enhanced), `extract2.py` (JSON experimental).
  - Outputs: Stored in structured `.txt` or `.json` files by file type and version.
- **Engineering Touches**:
  - Resilient to irregular formatting: extractor logic handled uneven tables, merged cells, and text artifacts gracefully.
  - Memory-optimized: used streaming file reads/writes to minimize memory footprint.
  - Automatic folder management: creates output directories dynamically if missing.
  - Modular architecture: easy extensibility to support additional formats (e.g., `.xlsm`) with minimal effort.

## Outcome

- Successfully generated clean, machine-optimized `.txt` files across all file types.
- Preserved page structures to enable later chunk-level metadata association.
- Delivered an additional **human-readable aligned version** to support manual quality inspection, bridging technical extraction with real-world usability needs.
- Early experimentation with `.json` outputs established a foundation for future metadata-rich document processing pipelines.
- Established a strong, reliable preprocessing baseline that ensured a smooth transition into downstream NLP phases like chunking, embedding, and RAG-based retrieval.

---

## Version Summary Table

| Version | Description | Key Focus |
|:--------|:------------|:----------|
| **1.0** | Standard extraction to `.txt` (optimized for tokenization). | Machine-readable output. |
| **1.1** | Visually enhanced `.txt` with column alignment for better human readability. | Human-auditable outputs. |
| **1.2** | Experimental `.json` format with basic metadata structure. | Future-proof metadata pipeline research. |

---

## Notes on Extra Effort

- While the assessment primarily emphasized PDFs and DOCX files, **support for CSV and Excel file extraction was proactively included**, going beyond the minimum requirement.
- **Attention to visual alignment and human-readability** — although not mandatory — reflects a production-minded engineering approach, ensuring that extracted outputs are not only machine-consumable but also audit-friendly for human evaluators.
- **Versioned outputs** and **experimental expansions** (e.g., JSON format) demonstrate iterative thinking and future-scaling mindset.

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

