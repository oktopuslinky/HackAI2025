import os
import torch
import fitz  # PyMuPDF
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

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
    try:
        # Sparse retriever: BM25
        bm25_retriever = BM25Retriever.from_documwhatents(documents)
        bm25_retriever.k = 5

        # Dense retriever: FAISS
        embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        faiss_store = FAISS.from_documents(documents, embeddings)
        faiss_retriever = faiss_store.as_retriever(search_kwargs={"k": 5})

        # Ensemble retriever combining sparse + dense
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.5, 0.5]
        )

        return ensemble_retriever
    except Exception as e:
        print(f"Error initializing retrievers: {e}")
        raise

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
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a financial expert. Using the following context, answer the question concisely and professionally. If the context is insufficient, provide a best-guess answer using financial reasoning.

Context:
{context}

Question:
{question}

Answer:"""
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        return qa_chain
    except Exception as e:
        print(f"Error building QA chain: {e}")
        raise

def main():
    pdf_path = "ltimindtree_annual_report.pdf"  # Your PDF file
    try:
        documents = chunk_pdf_text(pdf_path)
        retriever = build_ensemble_retriever(documents)
        llm = initialize_groq_llm()
        qa_chain = build_qa_chain(retriever, llm)

        print("\nWelcome to the Ensemble Financial Document QA System!")
        print("Type 'exit' to quit.\n")

        while True:
            query = input("You ➔ ").strip()
            if query.lower() == "exit":
                print("Goodbye!")
                break
            if not query:
                print("Please enter a valid question.\n")
                continue

            response = qa_chain.invoke({"query": query})
            print(f"\nAnswer: {response['result']}\n")
            # Uncomment this if you want to print some source doc snippets:
            # print("Sources:", [doc.page_content[:100] for doc in response['source_documents']])

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
