# turns pdf into image

import fitz  # PyMuPDF

# Open the PDF
pdf = fitz.open("ltimindtree_annual_report.pdf")
page = pdf[1]  # Page 2 (index 1)

# Step 1: Inspect path_data from get_drawings()
path_data = page.get_drawings()
print(f"Total drawings extracted: {len(path_data)}")
for idx, item in enumerate(path_data, 1):
    print(f"\nDrawing {idx}:")
    print(f"  Item keys: {list(item.keys())}")
    path_items = item.get('items', [])
    print(f"  Number of path items: {len(path_items)}")
    for sub_item in path_items:
        print(f"    Sub-item type: {sub_item[0]}")
        if sub_item[0] == 'pa':  # Path item
            path_obj = sub_item[1]
            try:
                d = path_obj.get_d()
                print(f"    Path data: {d[:100]}...")  # Truncate for readability
            except Exception as e:
                print(f"    Error getting path data: {e}")
        else:
            print(f"    Non-path item: {sub_item}")

# Step 2: Try rendering the page as an SVG
try:
    svg_content = page.get_svg_image(matrix=fitz.Matrix(1, 1))
    with open("page2_full.svg", "w") as f:
        f.write(svg_content)
    print("Saved entire page as page2_full.svg")
except Exception as e:
    print(f"Error generating SVG for page: {e}")

# Step 3: Save the page as a PNG for visual inspection
pix = page.get_pixmap(dpi=300)
pix.save("page2.png")
print("Saved page as page2.png")

# Close the PDF
pdf.close()

import cv2
import numpy as np
import os
try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = os.path.join(os.getcwd(), 'Tesseract-OCR', 'tesseract.exe')
    ocr_available = True
except ImportError:
    ocr_available = False
    print("Warning: pytesseract not installed. Text filtering will be less accurate.")

# Input PNG file
input_png = "page2.png"

# Step 1: Load the image
image = cv2.imread(input_png, cv2.IMREAD_UNCHANGED)
if image is None:
    print(f"Error: Could not load {input_png}")
    exit(1)

# Check for alpha channel
has_alpha = image.shape[2] == 4
print(f"Image {'has' if has_alpha else 'has no'} alpha channel")
gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY if has_alpha else cv2.COLOR_BGR2GRAY)

# Step 2: Preprocess the image
# Apply binary threshold to isolate objects
_, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

# Morphological operations to merge diagram parts and remove small noise
kernel = np.ones((10, 10), np.uint8)  # Larger kernel to connect diagram parts
binary = cv2.dilate(binary, kernel, iterations=2)  # Merge fragmented diagrams
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))  # Remove tiny noise

# Step 3: Find connected components
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
print(f"Found {num_labels - 1} connected components (excluding background)")

# Step 4: Extract and save diagram-like components
output_dir = "extracted_diagrams"
os.makedirs(output_dir, exist_ok=True)
valid_diagram_count = 0

for i in range(1, num_labels):  # Skip background (label 0)
    # Get bounding box and area
    x, y, w, h, area = stats[i]
    
    # Filter by area (diagrams are much larger than text)
    min_area = 10000  # Adjust for 300 DPI PNG (e.g., 10000 for diagrams, text is smaller)
    if area < min_area:
        print(f"Component {i}: Too small (area={area}), likely text, skipping.")
        continue
    
    # Filter by aspect ratio (diagrams are often square/rectangular, text is linear)
    aspect_ratio = w / h if h > 0 else float("inf")
    if not (0.2 < aspect_ratio < 5.0):
        print(f"Component {i}: Invalid aspect ratio ({aspect_ratio:.2f}), likely text, skipping.")
        continue
    
    # OCR to detect standalone text
    if ocr_available:
        region = image[y:y+h, x:x+w]
        text = pytesseract.image_to_string(region, config="--psm 6")
        if text.strip() and area < 50000:  # Allow text in large diagrams
            print(f"Component {i}: Contains standalone text ('{text[:50]}...'), area={area}, skipping.")
            continue
    
    # Extract the region
    mask = (labels == i).astype(np.uint8) * 255
    component = cv2.bitwise_and(image, image, mask=mask)
    
    # Crop to the bounding box
    component_cropped = component[y:y+h, x:x+w]
    
    # Create a transparent background
    if has_alpha:
        alpha = mask[y:y+h, x:x+w]
        component_cropped = cv2.merge((component_cropped[:, :, :3], alpha))
    else:
        alpha = np.ones((h, w), dtype=np.uint8) * 255
        alpha[mask[y:y+h, x:x+w] == 0] = 0
        component_cropped = cv2.merge((component_cropped[:, :, :3], alpha))
    
    # Save the diagram
    output_file = os.path.join(output_dir, f"diagram_{i}.png")
    cv2.imwrite(output_file, component_cropped)
    valid_diagram_count += 1
    print(f"Saved diagram {i} as {output_file} (x={x}, y={y}, w={w}, h={h}, area={area})")

# Step 5: Summary
print(f"\nProcessed {num_labels - 1} components.")
print(f"Saved {valid_diagram_count} diagrams to {output_dir}.")
if valid_diagram_count == 0:
    print("No diagrams were saved. Adjust area threshold, aspect ratio, or OCR settings.")


#########

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
    
    with open(f"extracted_diagrams/image_{i+1}.png", "wb") as f:
        f.write(img_data)

import pytesseract
from PIL import Image
import io

import os
import pytesseract
from PIL import Image
import cv2

# Set the Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = os.path.join(os.getcwd(), 'Tesseract-OCR', 'tesseract.exe')

def ocr_images_in_directory(directory):
    """
    Perform OCR on all PNG images in the specified directory and return a dictionary
    with image filenames as keys and extracted text as values.
    
    Args:
        directory (str): Path to the directory containing PNG images.
    
    Returns:
        dict: Dictionary with image filenames as keys and OCR-extracted text as values.
    """
    # Initialize the dictionary to store results
    ocr_results = {}
    
    # Ensure the directory exists
    if not os.path.exists(directory):
        print(f"Error: Directory {directory} does not exist.")
        return ocr_results
    
    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        # Process only PNG files
        if filename.lower().endswith('.png'):
            file_path = os.path.join(directory, filename)
            
            try:
                # Load the image using PIL
                image = Image.open(file_path)
                
                # Perform OCR
                text = pytesseract.image_to_string(image, config='--psm 6')
                
                # Store the result in the dictionary
                ocr_results[filename] = text.strip()
                
                print(f"Processed {filename}: {len(text)} characters extracted.")
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                ocr_results[filename] = ""
    
    return ocr_results

# Example usage
output_dir = "extracted_diagrams"
ocr_results = ocr_images_in_directory(output_dir)

# Print the results
print("\nOCR Results:")
for image_name, text in ocr_results.items():
    print(f"\nImage: {image_name}")
    print(f"Text: {text}")