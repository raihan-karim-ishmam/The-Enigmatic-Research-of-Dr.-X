# 📁 `input/` Folder

This folder is designed as the **dedicated input directory for Phase 5 (Translation)** and **Phase 6 (Summarization)**. These phases are run as **standalone modules**, so this folder is separated from the main `data/` directory used in the earlier pipeline stages.

---

## 🔎 Purpose

- Serves as the **entry point** for documents to be translated (Phase 5) or summarized (Phase 6).
- Enables independent testing and reusability of later phases without re-running the full extraction pipeline.

---

## 🔄 Supported Input Types

### For Summarization (Phase 6)
- Accepts raw `.pdf`, `.docx`, or `.txt` files
- Can include either:
  - Original documents (e.g., research papers, reports), or
  - Extracted text outputs from Phase 1 (already cleaned `.txt` files)

### For Translation (Phase 5)
- Accepts **only `.txt` files**
- Input files are expected to be **already extracted** and pre-cleaned
- Prevents re-extraction and ensures text is correctly formatted for LLM translation models

---

## ✅ Design Benefit

This decoupled structure prevents confusion between formats required by different phases, especially since:
- Summarization can operate directly on raw or extracted documents
- Translation strictly requires `.txt` format inputs

---

## 💡 Notes

- Do not mix file types unless both are supported by the phase you're running
- Outputs from this folder are saved to:
  - `translated/` (for Phase 5)
  - `summaries/` and `rouge_metrics/` (for Phase 6)

---

## 💻 Example Usage

```bash
# Phase 5 Translation
python translator.py 

# Phase 6 Summarization
python summarizer.py 
```

---

