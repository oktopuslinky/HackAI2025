import os
import torch
import fitz  # PyMuPDF
from dotenv import load_dotenv
import numpy as np
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
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Existing functions: extract_text_from_pdf, chunk_pdf_text, build_ensemble_retriever, 
# normalize_scores, compute_hybrid_scores, initialize_groq_llm, build_qa_chain (unchanged)
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
        return ensemble_retriever, bm25_retriever, faiss_retriever
    except Exception as e:
        print(f"Error initializing retrievers: {e}")
        raise

def normalize_scores(scores):
    if not scores:
        return scores
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [1.0] * len(scores)
    return [(score - min_score) / (max_score - min_score) for score in scores]

def compute_hybrid_scores(query, bm25_retriever, faiss_retriever, documents, weights=[0.5, 0.5]):
    try:
        # Get BM25 documents and scores
        bm25_docs = bm25_retriever.get_relevant_documents(query)
        if not bm25_docs:
            return []
        
        # Get FAISS documents and scores
        faiss_docs = faiss_retriever.get_relevant_documents(query)
        if not faiss_docs:
            return []
        
        # Get FAISS scores directly
        faiss_store = faiss_retriever.vectorstore
        query_embedding = faiss_store.embedding_function.embed_query(query)
        faiss_scores = faiss_store.index.search(np.array([query_embedding], dtype=np.float32), k=5)
        faiss_scores = faiss_scores[0].tolist()
        
        # Normalize sparse and dense RAG scores
        bm25_scores = [1.0 / (i + 1) for i in range(len(bm25_docs))]
        bm25_scores = normalize_scores(bm25_scores)
        faiss_scores = normalize_scores(faiss_scores)
        
        # Create a mapping of document content to scores
        doc_to_scores = {}
        
        # Add BM25 scores
        for doc, score in zip(bm25_docs, bm25_scores):
            doc_to_scores[doc.page_content] = {'bm25': score, 'faiss': 0.0}
        
        # Add FAISS scores
        for doc, score in zip(faiss_docs, faiss_scores):
            if doc.page_content in doc_to_scores:
                doc_to_scores[doc.page_content]['faiss'] = score
            else:
                doc_to_scores[doc.page_content] = {'bm25': 0.0, 'faiss': score}
        
        # Compute hybrid scores
        hybrid_scores = []
        for doc_content, scores in doc_to_scores.items():
            # hybrid score formula
            hybrid_score = weights[0] * scores['bm25'] + weights[1] * scores['faiss']
            hybrid_scores.append((doc_content, hybrid_score))
        
        # Sort for highest hybrid score
        hybrid_scores.sort(key=lambda x: x[1], reverse=True)
        return hybrid_scores
        
    except Exception as e:
        print(f"Error computing hybrid scores: {e}")
        return []

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

# def build_qa_chain(retriever, llm):
#     try:
#         prompt = PromptTemplate(
#             input_variables=["context", "question"],
#             template="""You are a financial expert. Using the following context, answer the question concisely and professionally. If the context is insufficient, provide a best-guess answer using financial reasoning.

# Context:
# {context}

# Question:
# {question}

# Answer:"""
#         )
#         qa_chain = RetrievalQA.from_chain_type(
#             llm=llm,
#             retriever=retriever,
#             chain_type="stuff",
#             chain_type_kwargs={"prompt": prompt},
#             return_source_documents=True
#         )
#         return qa_chain
#     except Exception as e:
#         print(f"Error building QA chain: {e}")
#         raise

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
            verbose=True,  # optional, can show intermediate steps
        )
        return qa_chain
    except Exception as e:
        print(f"Error building QA chain: {e}")
        raise


def main():
    pdf_path = "ltimindtree_annual_report.pdf"
    try:
        documents = chunk_pdf_text(pdf_path)
        ensemble_retriever, bm25_retriever, faiss_retriever = build_ensemble_retriever(documents)
        llm = initialize_groq_llm()
        qa_chain = build_qa_chain(ensemble_retriever, llm)

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

            
            # response = qa_chain.invoke({"query": query})
            # print(f"\nAnswer: {response['result']}\n")
            response = qa_chain({"question": query})
            print(f"\nAnswer: {response['answer']}\n")


            # Compute hybrid scores and display only the highest
            # hybrid_scores = compute_hybrid_scores(query, bm25_retriever, faiss_retriever, documents)
            # if hybrid_scores:  # Check if there are any scores
            #     top_doc, top_score = hybrid_scores[0]  # Get the document with the highest score
            #     print("Confidence Score:")
           
            #     print(f"Score: {top_score:.4f}")
            # else:
            #     print("No documents retrieved.\n")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()