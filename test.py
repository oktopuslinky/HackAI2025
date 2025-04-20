import os
import cv2
import numpy as np
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = os.path.join(os.getcwd(), 'Tesseract-OCR', 'tesseract.exe')

def polar_to_cartesian(image, center, radius, output_height=100):
    """
    Unwrap a circular region into a rectangular image using polar-to-Cartesian transformation.
    
    Args:
        image: Input image (grayscale or color).
        center: Tuple (x, y) of the circle's center.
        radius: Radius of the circular text region.
        output_height: Height of the output rectangular image.
    
    Returns:
        Unwrapped rectangular image.
    """
    # Define output dimensions
    output_width = int(2 * np.pi * radius)  # Circumference
    unwrapped = np.zeros((output_height, output_width), dtype=np.uint8)
    
    # Polar to Cartesian transformation
    for y in range(output_height):
        r = radius - (y * radius / output_height)  # From outer to inner radius
        for x in range(output_width):
            theta = x * 2 * np.pi / output_width  # Angle in radians
            src_x = int(center[0] + r * np.cos(theta))
            src_y = int(center[1] + r * np.sin(theta))
            
            # Ensure coordinates are within bounds
            if 0 <= src_y < image.shape[0] and 0 <= src_x < image.shape[1]:
                unwrapped[y, x] = image[src_y, src_x]
    
    return unwrapped

def detect_circular_text_region(image):
    """
    Detect the circular region likely containing text using Hough Circle Transform.
    
    Args:
        image: Input image (grayscale).
    
    Returns:
        Tuple (center, radius) of the detected circle, or None if not found.
    """
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (9, 9), 2)
    
    # Hough Circle Transform
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=100,
        param1=50,
        param2=30,
        minRadius=50,
        maxRadius=500
    )
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # Return the first detected circle (x, y, r)
        x, y, r = circles[0]
        return (x, y), r
    
    return None

def extract_circular_text(image_path):
    """
    Extract text from a circular region in an image.
    
    Args:
        image_path: Path to the input image.
    
    Returns:
        Extracted text or empty string if no circular text is found.
    """
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Could not load {image_path}")
        return ""
    
    # Detect circular text region
    result = detect_circular_text_region(image)
    if result is None:
        print(f"No circular text region detected in {image_path}")
        return ""
    
    center, radius = result
    print(f"Detected circle: center={center}, radius={radius}")
    
    # Unwrap the circular region
    unwrapped = polar_to_cartesian(image, center, radius)
    
    # Preprocess the unwrapped image
    _, binary = cv2.threshold(unwrapped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # Remove noise
    
    # Save intermediate unwrapped image for debugging (optional)
    debug_path = image_path.replace(".png", "_unwrapped.png")
    cv2.imwrite(debug_path, binary)
    print(f"Saved unwrapped image to {debug_path}")
    
    # Perform OCR
    pil_image = Image.fromarray(binary)
    text = pytesseract.image_to_string(pil_image, config='--psm 6')
    
    return text.strip()

def ocr_circular_text_in_directory(directory):
    """
    Perform OCR on circular text in all PNG images in the specified directory.
    
    Args:
        directory: Path to the directory containing PNG images.
    
    Returns:
        dict: Dictionary with image filenames as keys and OCR-extracted text as values.
    """
    ocr_results = {}
    
    if not os.path.exists(directory):
        print(f"Error: Directory {directory} does not exist.")
        return ocr_results
    
    for filename in os.listdir(directory):
        if filename.lower().endswith('.png'):
            file_path = os.path.join(directory, filename)
            print(f"\nProcessing {filename}...")
            text = extract_circular_text(file_path)
            ocr_results[filename] = text
            print(f"Extracted text: {text[:100]}..." if len(text) > 100 else f"Extracted text: {text}")
    
    return ocr_results

# Example usage
output_dir = "extracted_diagrams"
ocr_results = ocr_circular_text_in_directory(output_dir)

# Print summary
print("\nOCR Results for Circular Text:")
for image_name, text in ocr_results.items():
    print(f"\nImage: {image_name}")
    print(f"Text: {text[:100]}..." if len(text) > 100 else f"Text: {text}")

# Save results to a file (optional)
with open("circular_ocr_results.txt", "w", encoding="utf-8") as f:
    for image_name, text in ocr_results.items():
        f.write(f"Image: {image_name}\n")
        f.write(f"Text: {text}\n\n")
print("OCR results saved to circular_ocr_results.txt")