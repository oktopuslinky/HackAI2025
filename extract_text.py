from groq import Groq
import os
import base64
import numpy as np
from PIL import Image
import io

# Initialize Groq client (use environment variable for API key)
try:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
except Exception as e:
    print(f"Error initializing Groq client: {e}")
    exit(1)

# Directory containing images
directory = 'extracted_diagrams'

# Process each image in the directory
for image in os.listdir(directory):
    image_path = os.path.join(directory, image)
    if not image_path.lower().endswith('.png'):
        continue  # Skip non-PNG files
    print('IMAGE PATH:', image_path)

    try:
        # Open and process image for color extraction
        with Image.open(image_path) as img:
            # Convert to RGB
            img_rgb = img.convert("RGB")
            # Get image data as a numpy array
            pixels = np.array(img_rgb)
            # Reshape to a list of RGB tuples
            pixels = pixels.reshape(-1, 3)
            # Get unique colors
            unique_colors = np.unique(pixels, axis=0)
            # Print unique colors
            print("Unique Colors (RGB):")
            for color in unique_colors:
                print(f"RGB: {color}")

        # Read image for base64 encoding (for Groq API, if needed)
        with open(image_path, "rb") as image_file:
            image_encoding = base64.b64encode(image_file.read()).decode('utf-8')

        # Uncomment below if you want to use Groq API for text extraction
        '''
        try:
            completion = client.chat.completions.create(
                model="llama3-8b-8192",  # Replace with a valid Groq model
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": f"data:image/png;base64,{image_encoding}"
                            },
                            {"type": "text", "text": "Extract all the text visible in this image."}
                        ]
                    }
                ],
                temperature=0.5,
                top_p=1,
                stream=False,
                stop=None,
            )
            extracted_text = completion.choices[0].message.content
            print("Extracted Text:", extracted_text)
        except Exception as e:
            print(f"Error processing Groq API for {image_path}: {e}")
        '''

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
