from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import fitz  # PyMuPDF
import numpy as np
import torch

# LangChain modules
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=".", html=True), name="static")

# Save global variables for now
current_qa_chain = None
current_documents = None

# ----------------------
# Core Functions
# ----------------------

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    all_text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        all_text += f"\n--- Page {page_num + 1} ---\n{text}"
    all_text = ' '.join(all_text.split())
    return all_text

def chunk_pdf_text(pdf_path, chunk_size=2000, chunk_overlap=500):
    text = extract_text_from_pdf(pdf_path)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    text_chunks = text_splitter.split_text(text)
    documents = [Document(page_content=chunk, metadata={"source": pdf_path}) for chunk in text_chunks]
    return documents

def build_ensemble_retriever(documents):
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 5
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    faiss_store = FAISS.from_documents(documents, embeddings)
    faiss_retriever = faiss_store.as_retriever(search_kwargs={"k": 5})
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5]
    )
    return ensemble_retriever

def initialize_groq_llm():
    try:
        llm = ChatGroq(
            model_name="llama3-70b-8192",
            groq_api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.5,
            max_tokens=500
        )
        print("Groq LLM initialized successfully.")
        return llm
    except Exception as e:
        print(f"Error initializing Groq LLM: {e}")
        raise

def build_qa_chain(retriever, llm):
    try:
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            chain_type="stuff",
            verbose=True,
        )
        return qa_chain
    except Exception as e:
        print(f"Error building QA chain: {e}")
        raise

# ----------------------
# API Endpoints
# ----------------------

class QueryRequest(BaseModel):
    query: str

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global current_qa_chain, current_documents
    try:
        # Save file
        file_location = f"./{file.filename}"
        with open(file_location, "wb") as f:
            f.write(await file.read())

        # Process file
        documents = chunk_pdf_text(file_location)
        ensemble_retriever = build_ensemble_retriever(documents)
        llm = initialize_groq_llm()
        qa_chain = build_qa_chain(ensemble_retriever, llm)

        # Save for global session
        current_qa_chain = qa_chain
        current_documents = documents

        return JSONResponse(content={"message": "Extraction and setup complete!", "filename": file.filename})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/query")
async def query_pdf(query: QueryRequest):
    global current_qa_chain
    try:
        if not current_qa_chain:
            return JSONResponse(content={"error": "No document uploaded yet."}, status_code=400)

        response = current_qa_chain({"question": query.query})
        return JSONResponse(content={"answer": response['answer']})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    


# ----------------------
# To run server:
# uvicorn main:app --reload
# ----------------------
