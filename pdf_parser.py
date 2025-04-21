import pymupdf as fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch
from groq import Groq

hf_token = "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"

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    all_text = ""

    for page_num in range(len(doc)):
        page = doc.load_page(page_num) 
        text = page.get_text()
        all_text += f"\n--- Page {page_num + 1} ---\n{text}"

    return all_text

def chunk_raw_text(pdf_path, chunk_size=2000, chunk_overlap=500):
    text = extract_text_from_pdf(pdf_path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    text_chunks = text_splitter.split_text(text)
    documents = [Document(page_content=chunk, metadata={"source": pdf_path}) for chunk in text_chunks]
    return documents

def initialize_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name = 'sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs = {'device':'cpu'},
        encode_kwargs = {'normalize_embeddings':True}
    )
    return embeddings

def create_vector_store(documents, embeddings):
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

def dense_retrieval(query, vector_store, k=5):
    retriever = vector_store.as_retriever(search_kwargs={"k":k})
    relevant_docs = retriever.invoke(query)
    return relevant_docs

def perform_retrieval(documents): 
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )

    # Automatically convert chunks to embeddings
    vector_storage = FAISS.from_documents(documents, embedding_model)

    dense_retriever = vector_storage.as_retriever(search_kwargs={"k": 5})
    bm25_sparse_retriever = BM25Retriever.from_documents(documents)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, bm25_sparse_retriever],
        weights=[.5, .5]
    )



    cross_encoder_model_name = "cross-encoder/ms-marco-MiniLM-L6-v2"
    cross_encoder_tokenizer = AutoTokenizer.from_pretrained(cross_encoder_model_name)
    cross_encoder = AutoModelForSequenceClassification.from_pretrained(cross_encoder_model_name, token=hf_token)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cross_encoder.to(device)

    # Initialize the Groq client with API key from environment variable
    client = Groq(api_key="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")

    # Define the messages for the chat completion
    messages = []
    while True: 
        query = input("Enter your query: ")
        sub_message = messages.copy()
        sub_message += [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": f'''
                    Here is the question: {query}
                    If the query seems like a follow-up, rewrite it to include enough context from the previous question(s) such that it the resulting question is like a new question. Otherwise, just output the same question. Whatever you do, just output the question, nothing else.
                '''
            }
        ]
        chat_completion_subquery = client.chat.completions.create(
                messages=messages,
                model="llama-3.3-70b-versatile",  
                temperature=0.4,                  
                max_tokens=512,                   
                stream=False                      
            )
        
        query = chat_completion_subquery.choices[0].message.content
        initial_docs = ensemble_retriever.invoke(query)

        scores = []
        for doc in initial_docs:
            cross_encoder_tokenized_inputs = cross_encoder_tokenizer(
                query, 
                doc.page_content, 
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            with torch.no_grad():
                cross_encoder_output = cross_encoder(**cross_encoder_tokenized_inputs)
                score = cross_encoder_output.logits.softmax(dim=-1)[0][0].item()
            scores.append(score)
        

        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_docs = [initial_docs[i] for i in ranked_indices[:5]]

        llm_context = [str(doc.page_content) + " " for doc in top_docs]
        messages += [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": f'''
                    You are a financial analyst. 
                    Answer the query concisely using the provided context from LTIMindTree's annual report. If the context lacks sufficient information, say so and provide a general answer based on available data. 
                    Here is the context: {llm_context}
                    Here is the question: {query}
                '''
            }
        ]
        
        chat_completion = client.chat.completions.create(
                messages=messages,
                model="llama-3.3-70b-versatile",  
                temperature=0.4,                  
                max_tokens=512,                   
                stream=False                      
            )

        print("MESSAGE FROM LLAMA: ", chat_completion.choices[0].message.content)
  
  

pdf_path = "ltimindtree_annual_report.pdf"
extracted_text = extract_text_from_pdf(pdf_path)
documents = chunk_raw_text(pdf_path)
perform_retrieval(documents)
