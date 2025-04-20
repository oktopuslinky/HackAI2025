import os
import torch
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline

def build_ensemble_retriever(text_file_path: str):
    try:
        # Load and validate document
        if not os.path.exists(text_file_path):
            raise FileNotFoundError(f"Text file {text_file_path} not found.")
        loader = TextLoader(text_file_path)
        documents = loader.load()
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            add_start_index=True,
        )
        splits = text_splitter.split_documents(documents)
        
        # Initialize BM25 retrievers with different parameters
        bm25_retriever1 = BM25Retriever.from_documents(splits)
        bm25_retriever1.k = 4
        
        bm25_retriever2 = BM25Retriever.from_documents(splits)
        bm25_retriever2.k = 6
        
        # Initialize FAISS retriever for semantic search
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        faiss_store = FAISS.from_documents(splits, embeddings)
        faiss_retriever = faiss_store.as_retriever(search_kwargs={"k": 5})
        
        # Create ensemble retriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever1, bm25_retriever2, faiss_retriever],
            weights=[0.3, 0.3, 0.4]  # Higher weight for semantic search
        )
        
        return ensemble_retriever
    except Exception as e:
        print(f"Error initializing retriever: {e}")
        raise

def build_qa_chain(retriever):
    try:
        # Create prompt template to avoid "I don't know" and ensure professional responses
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a financial document expert. Based on the context below, provide a concise, accurate, and professional answer to the question. If the context lacks sufficient information, use your knowledge to provide a reasoned response or suggest a plausible answer based on financial principles.

Context: {context}
Question: {question}
Answer:"""
        )
        
        # Initialize HuggingFace pipeline with a more powerful model
        device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
        hf_pipeline = pipeline(
            "text2text-generation",
            model="google/flan-t5-large",
            max_length=512,
            temperature=0.3,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            device=device
        )
        
        local_llm = HuggingFacePipeline(pipeline=hf_pipeline)
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=local_llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True  # For debugging or transparency
        )
        
        return qa_chain
    except Exception as e:
        print(f"Error initializing QA chain: {e}")
        raise

def main():
    text_file_path = "extracted_text.txt"
    try:
        # Initialize retriever and QA chain
        retriever = build_ensemble_retriever(text_file_path)
        qa_chain = build_qa_chain(retriever)
        
        print("\nWelcome to the Enhanced Financial Document QA System!")
        print("Type 'exit' to quit\n")
        
        while True:
            query = input("Your question: ").strip()
            if query.lower() == "exit":
                print("Goodbye!")
                break
            if not query:
                print("Please enter a valid question.\n")
                continue
                
            # Get answer
            response = qa_chain.invoke({"query": query})
            print(f"\nAnswer: {response['result']}\n")
            # Optionally print source documents for transparency
            # print("Sources:", [doc.page_content[:100] for doc in response['source_documents']])
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
