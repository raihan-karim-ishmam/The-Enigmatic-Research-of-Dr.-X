# 📁 `data/` Folder

This directory serves as the **starting point** for the NLP pipeline. It contains the raw or preloaded files that feed into **Phase 1: Text Extraction** and initiate the Retrieval-Augmented Framework (RAF) flow, following uptill **Phase 4: RAG Q&A System Development**.

---

## 🔄 Pipeline Role

- Files in this folder are the **first inputs** of the system.
- These documents are processed by the extractors in **Phase 1**, which generate structured text for downstream tasks like chunking, embedding, and retrieval.
- All remaining pipeline scripts use the outputs from this phase — typically passed via `output/` folders — as their respective inputs.

---

## 📝 File Types

The folder may contain:

- `.pdf` — scientific articles, academic reports
- `.docx` — research manuscripts, project documents
- `.txt` — plain text files, notes, or exported chat logs

---

## ✅ Contents

This folder currently contains **all the original test documents** used to validate and benchmark the full pipeline during development — including multilingual texts, cancer research papers, and general academic documents.

These files form the **testbed for Phases 1–6**, and serve as a reproducible input set for anyone rerunning or extending the system.

---

## 🚨 Notes

- Do **not** rename or move files during an active run — the pipeline expects stable paths.
- Subfolders may be created dynamically for organized storage (e.g., per phase, per doc type).
- This folder should not store final outputs or summary data — only **starting material** for extraction.

---

## 💻 Example Usage

```bash
# Run Phase 1 extract script
python extract.py 
```

---

