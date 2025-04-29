# Phase 1 - Version 1.0 - Conversion to text (optimized for tokens with no addiotnal spaces) with [PAGE X] markers and ensuring redability 

import os
import docx  # For reading Word files
import PyPDF2  # For reading PDF files
import pandas as pd  # For reading Excel and CSV files

# Folders where input files are located and outputs will be saved
INPUT_FOLDER = "data"
OUTPUT_FOLDER = "output"

# Create the output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -------- FUNCTIONS TO EXTRACT TEXT -------- #

# Extract text from .docx (Word) files, simulate pages
def extract_docx(file_path):
    doc = docx.Document(file_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    text = ""
    for idx, para in enumerate(paragraphs):
        if idx % 50 == 0:
            page_num = idx // 50 + 1
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

# Extract text from Excel files, simulate pages
def extract_excel(file_path):
    xls = pd.ExcelFile(file_path)
    text = ""
    for sheet in xls.sheet_names:
        df = xls.parse(sheet).dropna(how="all").fillna("")
        rows = df.astype(str).values.tolist()
        for idx, row in enumerate(rows):
            if idx % 30 == 0:
                page_num = idx // 30 + 1
                text += f"\n\n[PAGE {page_num}]\n\n"
            text += "\t".join(row) + "\n"
    return text

# Extract text from CSV files, simulate pages
def extract_csv(file_path):
    df = pd.read_csv(file_path).dropna(how="all").fillna("")
    rows = df.astype(str).values.tolist()
    text = ""
    for idx, row in enumerate(rows):
        if idx % 30 == 0:
            page_num = idx // 30 + 1
            text += f"\n\n[PAGE {page_num}]\n\n"
        text += "\t".join(row) + "\n"
    return text

# -------- MAIN DISPATCH FUNCTION -------- #

# Choose the extraction method based on file extension
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
            out_name = os.path.splitext(filename)[0] + ".txt"
            output_path = os.path.join(OUTPUT_FOLDER, out_name)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)
        else:
            print(f"⚠️ Skipped unsupported file: {filename}")

    except Exception as e:
        print(f"❌ Error with {filename}: {e}")
