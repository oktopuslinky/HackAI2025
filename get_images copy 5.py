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