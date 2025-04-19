import fitz  # PyMuPDF

def convert_pdf_to_text(pdf_path, output_file):
    # Open the PDF file
    doc = fitz.open(pdf_path)
    
    # Initialize empty text
    full_text = ""
    
    # Iterate through each page
    for page in doc:
        # Extract text from the page
        text = page.get_text()
        # Add page text to full text without newline
        full_text += text + " "
    
    # Remove extra spaces and line breaks
    full_text = ' '.join(full_text.split())
    
    # Write the extracted text to a file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(full_text)
    
    print(f"PDF has been converted to text and saved to {output_file}")

if __name__ == "__main__":
    pdf_path = "ltimindtree_annual_report.pdf"  # Your PDF file
    output_file = "extracted_text.txt"  # Output text file
    
    convert_pdf_to_text(pdf_path, output_file) 