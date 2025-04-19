import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import json
import torch
from transformers import BartForConditionalGeneration, BartTokenizer, pipeline
import re
import os
import openai

import json

# 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

# === OCR SETTINGS ===
PDF_PATH = "ltimindtree_annual_report.pdf"
TESSERACT_LANG = "eng"
pytesseract.pytesseract.tesseract_cmd = os.path.join(os.getcwd(), 'Tesseract-OCR', 'tesseract.exe')


# === Load ReBEL Model ===
model = BartForConditionalGeneration.from_pretrained("Babelscape/rebel-large")
tokenizer = BartTokenizer.from_pretrained("Babelscape/rebel-large")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# === 1. Convert PDF to Images ===
def pdf_to_images(pdf_path):
    poppler_path = os.path.join(os.getcwd(), 'poppler-24.08.0', 'Library', 'bin')
    return convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path, first_page=1, last_page=1)

# === 2. Run OCR on Each Page ===
def extract_text_from_images(images):
    pages_text = []
    for i, img in enumerate(images):
        print(f"Running OCR on page {i + 1}...")
        text = pytesseract.image_to_string(img, lang=TESSERACT_LANG)
        pages_text.append(text)
        print('LENGTH:', len(text.split()))
    
    #print(pages_text)
    return pages_text

def clean_text(text):
    sentences = ''
    from groq import Groq

    client = Groq(
        api_key="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",
        #api_url="https://api.groq.com/v1"
    )
    completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": "I have extracted text from a pdf using OCR. I want you to turn this extracted text into line separated human-readable sentences. Do this for all of then given text, not just certain parts. The output should be just these sentences, and nothing else. These sentences should be in a factual statement form. The OCR text is: " + " ".join(text)
            }
        ],
        temperature=1,
        #max_completion_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )
    
    for chunk in completion:
        content = chunk.choices[0].delta.content or ""
        print('--', content)
        sentences += content
        #print(chunk.choices[0].delta.content or "", end="")
    
    get_triplets(sentences)
    #print(sentences)
    return sentences

def get_triplets(text):
    sentences = ''
    from groq import Groq

    client = Groq(
        api_key="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",
        #api_url="https://api.groq.com/v1"
    )
    completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": "I have a list of sentence statements. I want you to convert each sentence into triplets for use with rebel. This is the text: " + " ".join(text)
            }
        ],
        temperature=1,
        #max_completion_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )
    
    for chunk in completion:
        content = chunk.choices[0].delta.content or ""
        #print('--', content)
        sentences += content
        #print(chunk.choices[0].delta.content or "", end="")
    
    print(sentences)
    return sentences

# === 4. ReBEL: Extract Triplets ===

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large").to('cuda' if torch.cuda.is_available() else 'cpu')
model.eval()

def extract_triplets(text):
    text_chunks = split_text(text, max_len=400)
    triplets = []
    for chunk in text_chunks:
        inputs = tokenizer(chunk, return_tensors="pt", max_length=512, truncation=True).to(model.device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, num_beams=5, max_length=512)
        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        print("Model output:", decoded)  # 👈 Check this!
        chunk_triplets = parse_triplets(decoded)
        triplets.extend(chunk_triplets)
    return triplets

def split_text(text, max_len=400):
    sentences = [s.strip() for s in text.replace('\n', ' ').split('.') if len(s.strip()) > 0]
    chunks = []
    current = ""
    for s in sentences:
        if len(current) + len(s) < max_len:
            current += s + ". "
        else:
            chunks.append(current.strip())
            current = s + ". "
    if current:
        chunks.append(current.strip())
    return chunks

def parse_triplets(generated_text):
    triplets = []
    
    # Split output into lines (assume each line = 1 triplet)
    lines = generated_text.strip().split("\n")
    
    for line in lines:
        parts = line.strip().split("  ")  # double space delimiter
        parts = [p.strip() for p in parts if p.strip()]
        
        if len(parts) == 3:
            triplets.append({
                "head": parts[0],
                "relation": parts[2],
                "tail": parts[1]
            })
    
    return triplets

def extract_numerical_facts(text):
    patterns = [
        r"([\w\s&]+?)\s+(?:was|is|were|are|reported as|reported|equals|equal to|stood at|amounted to|rated at|at|of)\s+(INR|USD|\$)?([\d,]+(?:\.\d+)?)(\s*(million|billion|GJ|ML|%|percent|days|people|employees|years|rating))?",
    ]
    
    triplets = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            subject = match[0].strip()
            currency_or_symbol = match[1].strip()
            value = match[2].strip()
            unit = match[4].strip()
            tail = f"{currency_or_symbol}{value} {unit}".strip()
            triplets.append({
                "head": subject,
                "relation": "has value",
                "tail": tail
            })
    return triplets




# === 5. Build JSON Graph ===
def build_json_graph(triplets):
    nodes = {}
    edges = []
    for t in triplets:
        for node in [t["head"], t["tail"]]:
            if node not in nodes:
                nodes[node] = {"id": node}
        edges.append({
            "source": t["head"],
            "target": t["tail"],
            "relation": t["relation"]
        })
    return {
        "nodes": list(nodes.values()),
        "edges": edges
    }


# === 6. Run the Pipeline ===
def process_scanned_pdf(pdf_path, output_json="graph.json"):
    images = pdf_to_images(pdf_path)
    raw_text_pages = extract_text_from_images(images)
    all_triplets = []

    for page_text in raw_text_pages:
        #cleaned = reformat_metrics_to_sentences(page_text)
        cleaned = clean_text(page_text)
        #print(cleaned)
        triplets = extract_triplets(cleaned)
        print('REGULAR TRIPLETS:',triplets)
        numerical_triplets = extract_numerical_facts(cleaned)
        print('NUMERICAL TRIPLETS:',numerical_triplets)
        
        all_triplets.extend(triplets)
        all_triplets.extend(numerical_triplets)

    graph = build_json_graph(all_triplets)
    print("Graph:", graph)
    '''with open(output_json, "w") as f:
        json.dump(graph, f, indent=2)'''

    print(f"Saved graph to {output_json}")
    return graph

# === 7. Run It ===
if __name__ == "__main__":
    #images = pdf_to_images(PDF_PATH)
    #raw_text_pages = extract_text_from_images(images)

    #print(raw_text_pages)
    


    process_scanned_pdf(PDF_PATH)
