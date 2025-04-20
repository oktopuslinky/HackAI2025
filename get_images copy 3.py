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