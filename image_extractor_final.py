def get_descriptions(filename):
    '''
    gets descriptions of important images in the pdf
    '''

    import fitz  # PyMuPDF

    # Open the PDF
    pdf = fitz.open(filename)

    for page_number in range(len(pdf)):
        page = pdf[page_number]
        image_count = 0
        print(f"Processing page {page_number + 1}...")
        
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

        try:
            svg_content = page.get_svg_image(matrix=fitz.Matrix(1, 1))
            with open(f"page{page_number+1}_full.svg", "w") as f:
                f.write(svg_content)
            print(f"Saved entire page as page{page_number+1}_full.svg")
        except Exception as e:
            print(f"Error generating SVG for page: {e}")

        # Step 3: Save the page as a PNG for visual inspection
        pix = page.get_pixmap(dpi=300)
        pix.save(f"page{page_number+1}.png")
        print(f"Saved page as page{page_number+1}.png")
        
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
        input_png = f"page{page_number+1}.png"

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
            valid_diagram_count += 1
            output_file = os.path.join(output_dir, f"{page_number+1}_{valid_diagram_count}.png")
            cv2.imwrite(output_file, component_cropped)
            #print(f"Saved diagram {i} as {output_file} (x={x}, y={y}, w={w}, h={h}, area={area})")

        import fitz  # PyMuPDF
        import os
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = os.path.join(os.getcwd(), 'Tesseract-OCR', 'tesseract.exe')
        # Function to extract images from a PDF
        def extract_images_from_pdf(pdf_path):
            # Open the PDF file
            doc = fitz.open(pdf_path)
            
            img_list = page.get_images(full=True)
            for img_index, img in enumerate(img_list):
                xref = img[0]
                image = page._document.extract_image(xref)
                image_bytes = image["image"]
                image_filename = os.path.join(output_dir, f"{page_number+1}_{img_index + image_count}.png")
                with open(image_filename, "wb") as f:
                    f.write(image_bytes)

    #############

    import os
    import numpy as np
    from PIL import Image
    from webcolors import rgb_to_name
    import requests
    import pytesseract

    # Minimal fallback color dictionary (basic colors for API failure)
    FALLBACK_COLORS = {
        'black': '#000000',
        'white': '#ffffff',
        'red': '#ff0000',
        'green': '#00ff00',
        'blue': '#0000ff',
        'yellow': '#ffff00',
        'cyan': '#00ffff',
        'magenta': '#ff00ff'
    }

    # Function to get color name from TheColorAPI
    def get_color_name_api(rgb):
        r, g, b = rgb
        try:
            response = requests.get(f"https://www.thecolorapi.com/id?rgb={r},{g},{b}", timeout=5)
            response.raise_for_status()
            data = response.json()
            return data.get("name", {}).get("value", "Unknown")
        except requests.RequestException as e:
            print(f"API request failed for RGB {rgb}: {e}")
            return None

    # Fallback function for closest color name
    def closest_color_fallback(rgb):
        try:
            # Try exact match with webcolors
            return rgb_to_name(rgb, spec='css3')
        except ValueError:
            # Find closest color by Euclidean distance
            min_distance = float('inf')
            closest_name = None
            r, g, b = rgb

            for name, hex_color in FALLBACK_COLORS.items():
                hr, hg, hb = tuple(int(hex_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                distance = ((r - hr) ** 2 + (g - hg) ** 2 + (b - hb) ** 2) ** 0.5
                if distance < min_distance:
                    min_distance = distance
                    closest_name = name

            return closest_name if closest_name else "Unknown"

    # Combined function to get color name (API first, then fallback)
    def closest_color(rgb):
        color_name = get_color_name_api(rgb)
        if color_name and color_name != "Unknown":
            return color_name
        return closest_color_fallback(rgb)

    # Directory containing images
    directory = 'extracted_diagrams'

    image_to_colors = {}

    # Process each image in the directory
    for image in os.listdir(directory):
        image_path = os.path.join(directory, image)
        if not image_path.lower().endswith('.png'):
            continue
        print('IMAGE PATH:', image_path)
        if image not in image_to_colors:
            image_to_colors[image] = []

        try:
            # Open and process image
            with Image.open(image_path) as img:
                # Check if image has an alpha channel
                has_alpha = img.mode == 'RGBA'
                transparent_pixels = 0
                if has_alpha:
                    # Convert to RGBA and resize
                    img_rgba = img.convert("RGBA").resize((100, 100))
                    pixels_rgba = np.array(img_rgba)
                    # Count fully transparent pixels
                    transparent_pixels = np.sum(pixels_rgba[:, :, 3] == 0)
                    print(f"Number of fully transparent pixels: {transparent_pixels}")

                    # Filter fully opaque pixels
                    opaque_pixels = pixels_rgba[pixels_rgba[:, :, 3] == 255][:, :3]
                    if len(opaque_pixels) == 0:
                        print("No fully opaque pixels found in the image.")
                        continue
                else:
                    # No alpha channel, use all pixels
                    img_rgb = img.convert("RGB").resize((100, 100))
                    opaque_pixels = np.array(img_rgb).reshape(-1, 3)

                # Get unique colors and counts
                unique_colors, counts = np.unique(opaque_pixels, axis=0, return_counts=True)
                # Sort by count and get top 2
                top_indices = np.argsort(counts)[-2:][::-1]
                top_colors = unique_colors[top_indices]
                top_counts = counts[top_indices]

                # Print top 2 colors
                print("Top 2 Most Common Colors (RGB, Name, Count):")
                for i, (color, count) in enumerate(zip(top_colors, top_counts)):
                    r, g, b = color
                    color_name = closest_color((r, g, b))
                    print(f"{image} Color {i+1}: RGB: {color}, Color Name: {color_name}, Count: {count}")
                    image_to_colors[image].append(color_name)

        

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    print(image_to_colors)

    skip_list = ['White', 'Black', 'Beige', 'gray', 'Athens Gray', 'Coconut', 'French Gray', 'Platinum', 'White Smoke', 'Gainsboro', 'Light Gray', 'Lavender Gray', 'Light Gray', 'Light Silver', 'Silver Chalice', 'Silver Sand', 'Bridal Heath', 'Quarter Pearl Lusta', 'Nobel', 'Black White']
    pytesseract.pytesseract.tesseract_cmd = os.path.join(os.getcwd(), 'Tesseract-OCR', 'tesseract.exe')

    images_with_text = []

    for image_name, colors in image_to_colors.items():
        if colors[0] not in skip_list and colors[1] not in skip_list:
            # try ocr
            filepath = os.path.join(directory, image_name)
            try:
                img = Image.open(filepath)
                text = pytesseract.image_to_string(img, config='--psm 6')
                if text.strip():
                    images_with_text.append(image_name)
                    print(f"Image {image_name} contains text: {text.strip()}")
            except Exception as e:
                print(f"Error during OCR for {image_name}: {e}")


    print("Images with text:", images_with_text)

    image_descriptions = []

    for image in images_with_text:
        import openai
        from openai import OpenAI
        import base64
        import os

        # Set your OpenAI API key
        client = OpenAI(
            api_key='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'
        )
        # Path to your image using os
        image_path = os.path.join(directory, image)  # Replace with your image path

        # Encode the image
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found at: {image_path}")
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")

            page, image_num = (image.split('.'))[0].split('_')
            cur_description = f"This image is from page {page} of the given pdf. "
            # Prepare the API request
            prompt = 'Please tell me the text in this image, and please describe this image and what it means'
            response = client.responses.create(
                model="gpt-4o-mini",
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": f"data:image/png;base64,{base64_image}"},
                        ],
                    }
                ],
            )

            # Extract the description
            cur_description += response.output_text
            image_descriptions.append(cur_description)
            print("Image Description:", cur_description)

        except Exception as e:
            print("Error:", str(e))

    print("Image Descriptions:")
    for desc in image_descriptions:
        print(desc)
        print("===" * 20)