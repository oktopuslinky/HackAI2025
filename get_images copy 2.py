import cv2
import numpy as np
import os

# Input PNG file
input_png = "page2.png"

# Step 1: Load the image
image = cv2.imread(input_png, cv2.IMREAD_UNCHANGED)
if image is None:
    print(f"Error: Could not load {input_png}")
    exit(1)

# Check if the image has an alpha channel; if not, use RGB
if image.shape[2] == 4:
    print("Image has alpha channel")
    gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
else:
    print("Image has no alpha channel")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 2: Preprocess the image
# Apply a binary threshold to create a mask of objects (adjust threshold as needed)
_, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
# Remove small noise with morphological operations
kernel = np.ones((5, 5), np.uint8)
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

# Step 3: Find connected components
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
print(f"Found {num_labels - 1} connected components (excluding background)")

# Step 4: Extract and save each component
output_dir = "extracted_images"
os.makedirs(output_dir, exist_ok=True)
valid_image_count = 0

for i in range(1, num_labels):  # Skip background (label 0)
    # Get the bounding box of the component
    x, y, w, h, area = stats[i]
    
    # Filter out small components (adjust area threshold as needed)
    if area < 500:  # Ignore components smaller than 500 pixels
        print(f"Component {i}: Too small (area={area}), skipping.")
        continue
    
    # Extract the region
    mask = (labels == i).astype(np.uint8) * 255
    component = cv2.bitwise_and(image, image, mask=mask)
    
    # Crop to the bounding box
    component_cropped = component[y:y+h, x:x+w]
    
    # Create a transparent background if the original image has alpha
    if image.shape[2] == 4:
        alpha = mask[y:y+h, x:x+w]
        component_cropped = cv2.merge((component_cropped[:, :, :3], alpha))
    else:
        # Add an alpha channel (fully opaque)
        alpha = np.ones((h, w), dtype=np.uint8) * 255
        alpha[mask[y:y+h, x:x+w] == 0] = 0
        component_cropped = cv2.merge((component_cropped[:, :, :3], alpha))
    
    # Save the component as a PNG
    output_file = os.path.join(output_dir, f"image_{i}.png")
    cv2.imwrite(output_file, component_cropped)
    valid_image_count += 1
    print(f"Saved component {i} as {output_file} (x={x}, y={y}, w={w}, h={h}, area={area})")

# Step 5: Alternative - Contour-based extraction
print("\nTrying contour-based extraction...")
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Found {len(contours)} contours")

for i, contour in enumerate(contours, 1):
    # Get the bounding box of the contour
    x, y, w, h = cv2.boundingRect(contour)
    
    # Filter out small contours
    area = cv2.contourArea(contour)
    if area < 500:
        print(f"Contour {i}: Too small (area={area}), skipping.")
        continue
    
    # Create a mask for the contour
    mask = np.zeros_like(binary)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    
    # Extract the region
    component = cv2.bitwise_and(image, image, mask=mask)
    component_cropped = component[y:y+h, x:x+w]
    
    # Create a transparent background
    if image.shape[2] == 4:
        alpha = mask[y:y+h, x:x+w]
        component_cropped = cv2.merge((component_cropped[:, :, :3], alpha))
    else:
        alpha = np.ones((h, w), dtype=np.uint8) * 255
        alpha[mask[y:y+h, x:x+w] == 0] = 0
        component_cropped = cv2.merge((component_cropped[:, :, :3], alpha))
    
    # Save the contour as a PNG
    output_file = os.path.join(output_dir, f"contour_{i}.png")
    cv2.imwrite(output_file, component_cropped)
    valid_image_count += 1
    print(f"Saved contour {i} as {output_file} (x={x}, y={y}, w={w}, h={h}, area={area})")

# Summary
print(f"\nProcessed {num_labels - 1} components and {len(contours)} contours.")
print(f"Saved {valid_image_count} images in total to {output_dir}.")
if valid_image_count == 0:
    print("No valid images were saved. Adjust thresholds or check the PNG content.")