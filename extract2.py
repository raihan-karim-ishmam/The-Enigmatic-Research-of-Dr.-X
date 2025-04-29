# Phase 1 - Version 1.2 - Conversion to structured JSON with [PAGE X] markers (not readability adapted)

import os
import json
import docx  # For reading Word files
import PyPDF2  # For reading PDF files
import pandas as pd  # For reading Excel and CSV files

# Folders where input files are located and outputs will be saved
INPUT_FOLDER = "data"
OUTPUT_FOLDER = "output_json"

# Create the output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Parameters for page simulation
LINES_PER_PAGE_WORD = 50  # Simulate 1 page per 50 lines for Word
ROWS_PER_PAGE_EXCEL = 30  # Simulate 1 page per 30 rows for Excel/CSV

# -------- FUNCTIONS TO EXTRACT TEXT -------- #

# Extract text from .docx (Word) files, simulate pages
def extract_docx(file_path):
    doc = docx.Document(file_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    text = ""
    for idx, para in enumerate(paragraphs):
        if idx % LINES_PER_PAGE_WORD == 0:
            page_num = idx // LINES_PER_PAGE_WORD + 1
            text += f"\n\n[PAGE {page_num}]\n\n"
        text += para + "\n"
    return text

# Extract text from .pdf files, real pages
def extract_pdf(file_path):
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            text += f"\n\n[PAGE {page_num + 1}]\n\n"
            text += page_text + "\n"
    return text

# Extract text from Excel files (.xls, .xlsx, .xlsm) with simulated pages
def extract_excel(file_path):
    xls = pd.ExcelFile(file_path)
    text = ""
    for sheet in xls.sheet_names:
        df = xls.parse(sheet).dropna(how="all").fillna("")
        rows = df.astype(str).values.tolist()
        for idx, row in enumerate(rows):
            if idx % ROWS_PER_PAGE_EXCEL == 0:
                page_num = idx // ROWS_PER_PAGE_EXCEL + 1
                text += f"\n\n[PAGE {page_num}]\n\n"
            text += "\t".join(row) + "\n"
    return text

# Extract text from CSV files with simulated pages
def extract_csv(file_path):
    df = pd.read_csv(file_path).dropna(how="all").fillna("")
    rows = df.astype(str).values.tolist()
    text = ""
    for idx, row in enumerate(rows):
        if idx % ROWS_PER_PAGE_EXCEL == 0:
            page_num = idx // ROWS_PER_PAGE_EXCEL + 1
            text += f"\n\n[PAGE {page_num}]\n\n"
        text += "\t".join(row) + "\n"
    return text

# -------- MAIN DISPATCH FUNCTION -------- #

# Choose the correct extraction function based on file type
def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".docx":
        return extract_docx(file_path)
    elif ext == ".pdf":
        return extract_pdf(file_path)
    elif ext in [".xlsx", ".xls", ".xlsm"]:
        return extract_excel(file_path)
    elif ext == ".csv":
        return extract_csv(file_path)
    else:
        return None  # Unsupported file type

# -------- MAIN LOOP -------- #

# Loop through all files in the input folder
for filename in os.listdir(INPUT_FOLDER):
    file_path = os.path.join(INPUT_FOLDER, filename)
    if not os.path.isfile(file_path):
        continue

    try:
        print(f"Processing: {filename}")
        content = extract_text(file_path)

        if content:
            json_data = {
                "filename": filename,
                "content": content.strip()
            }
            out_name = os.path.splitext(filename)[0] + ".json"
            output_path = os.path.join(OUTPUT_FOLDER, out_name)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
        else:
            print(f"⚠️ Skipped unsupported file: {filename}")

    except Exception as e:
        print(f"❌ Error with {filename}: {e}")
