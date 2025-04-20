import fitz  # PyMuPDF
import os
import pytesseract
pytesseract.pytesseract.tesseract_cmd = os.path.join(os.getcwd(), 'Tesseract-OCR', 'tesseract.exe')
# Function to extract images from a PDF
def extract_images_from_pdf(pdf_path):
    # Open the PDF file
    doc = fitz.open(pdf_path)
    
    image_list = []
    
    # Iterate through the pages of the PDF
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        
        # Get the list of images on the page
        img_list = page.get_images(full=True)
        
        for img_index, img in enumerate(img_list):
            xref = img[0]
            image = doc.extract_image(xref)
            image_bytes = image["image"]  # This is the image as a byte array
            
            image_list.append(image_bytes)
    
    return image_list

# Example usage
pdf_path = "ltimindtree_annual_report.pdf"
images = extract_images_from_pdf(pdf_path)

# Save images to files (optional)
for i, img_data in enumerate(images):
    with open(f"image_{i+1}.png", "wb") as f:
        f.write(img_data)



import pytesseract
from PIL import Image
import io

# Run OCR on a PIL image (from byte data)
def ocr_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))  # Convert byte data to Image object
    text = pytesseract.image_to_string(image)
    return text

# Example usage
for i, img_data in enumerate(images):
    text = ocr_image(img_data)
    print(f"Text from image {i+1}:\n{text}\n")