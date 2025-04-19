from pdf2image import convert_from_path
import easyocr
import os
import numpy as np

# === CONFIG ===
pdf_path = "ltimindtree_annual_report.pdf"
output_text_file = "ocr_output.txt"
language_list = ['en']  # You can add more languages if needed, e.g. ['en', 'fr']
poppler_path = os.path.join(os.getcwd(), 'poppler-24.08.0', 'Library', 'bin')

# === STEP 1: Convert PDF pages to images ===
print("Converting PDF pages to images...")
pages = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path, first_page=1, last_page=2)

# === STEP 2: Initialize OCR Reader ===
print("Initializing EasyOCR reader...")
reader = easyocr.Reader(language_list)

# === STEP 3: Run OCR and write to file ===
print(f"Running OCR and saving output to '{output_text_file}'...")
with open(output_text_file, "w", encoding="utf-8") as f:
    for i, page in enumerate(pages):
        f.write(f"\n--- Page {i+1} ---\n")
        print(f"Processing Page {i+1}...")
        
        # Convert PIL image to NumPy array
        page_np = np.array(page)
        
        # Perform OCR on the image
        results = reader.readtext(page_np)
        
        # Write OCR results to file
        for (bbox, text, confidence) in results:
            f.write(f"{text} (confidence: {confidence:.2f})\n")

print("✅ OCR completed and saved!")
