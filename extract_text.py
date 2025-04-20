from groq import Groq
import os
import base64

client = Groq(
    api_key="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",
    #api_url="https://api.groq.com/v1"
)

directory = 'extracted_diagrams'
for image in os.listdir(directory):
    image_path = os.path.join(directory, image)
    if not image_path.lower().endswith('.png'):
        continue  # Skip non-PNG files
    print('IMAGE PATH: ', image_path)
    with open(image_path, "rb") as image_file:
        image_encoding = base64.b64encode(image_file.read()).decode('utf-8')
    
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
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
            #max_completion_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
        )

        extracted_text = completion.choices[0].message.content
        print("Extracted Text:", extracted_text)