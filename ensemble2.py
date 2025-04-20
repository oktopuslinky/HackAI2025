import os
import torch
import pymupdf as fitz  
from dotenv import load_dotenv
import numpy as np
import uuid
from sentence_transformers import CrossEncoder
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        all_text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            all_text += f"\ gobern--- Page {page_num + 1} ---\n{text}"
        all_text = ' '.join(all_text.split())
        return all_text
    except Exception as e:
        raise Exception(f"Failed to extract text from PDF: {e}")

def chunk_pdf_text(pdf_path, chunk_size=2000, chunk_overlap=500):
    try:
        text = extract_text_from_pdf(pdf_path)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        text_chunks = text_splitter.split_text(text)
        # Assign unique IDs to each document (problem with re ranking)
        documents = [
            Document(
                page_content=chunk,
                metadata={"source": pdf_path, "chunk_id": str(uuid.uuid4())}
            ) for chunk in text_chunks
        ]
        return documents
    except Exception as e:
        raise Exception(f"Failed to chunk PDF text: {e}")

def build_ensemble_retriever(documents):
    try:
        
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 10  # Increased for re-ranking

        
        embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        faiss_store = FAISS.from_documents(documents, embeddings, ids=[doc.metadata["chunk_id"] for doc in documents])
        faiss_retriever = faiss_store.as_retriever(search_kwargs={"k": 10})  # Increased for re-ranking

        
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.5, 0.5]
        )
        return ensemble_retriever, bm25_retriever, faiss_retriever
    except Exception as e:
        raise Exception(f"Error initializing retrievers: {e}")

def initialize_cross_encoder():
    try:
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')
        print("Cross-encoder initialized yay!")
        return cross_encoder
    except Exception as e:
        raise Exception(f"Error: {e}")

def deduplicate_documents(documents):
    # Basically removing duplicate documents based on page_content and assigning unique IDs.
    seen_content = set()
    unique_docs = []
    for doc in documents:
        content = doc.page_content
        if content not in seen_content:
            seen_content.add(content)
            # Assigning a new unique ID to avoid conflicts
            doc.metadata["chunk_id"] = str(uuid.uuid4())
            unique_docs.append(doc)
    return unique_docs

def rerank_documents(query, documents, cross_encoder, top_k=5):
    try:
        # Pairs for cross-encoder
        pairs = [[query, doc.page_content] for doc in documents]

        scores = cross_encoder.predict(pairs)
        # Pair documents with their scores and sort
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        # Return top_k docs
        return [doc for doc, score in scored_docs[:top_k]]
    except Exception as e:
        print(f"Error during re-ranking: {e}")
        return documents  # Fallback to original docs

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
            groq_api_key="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",
            temperature=0.5,
            max_tokens=500
        )
        print("Groq LLM initialized successfully.")
        return llm
    except Exception as e:
        raise Exception(f"Error initializing Groq LLM: {e}")

def build_qa_chain(retriever, llm):
    try:
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a financial expert. Using the following context, answer the question concisely and professionally. If the context is insufficient, provide a best-guess answer using financial reasoning.

Context:
{context}

Question:
{question}

Answer:"""
        )
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            chain_type="stuff",
            combine_docs_chain_kwargs={"prompt": prompt},
            verbose=True,
        )
        return qa_chain
    except Exception as e:
        raise Exception(f"Error building QA chain: {e}")

def main():
    pdf_path = "ltimindtree_annual_report.pdf"
    try:

        
        documents = chunk_pdf_text(pdf_path)
        if not documents:
            raise Exception("No documents extracted from PDF.")

        ensemble_retriever, bm25_retriever, faiss_retriever = build_ensemble_retriever(documents)
        llm = initialize_groq_llm()
        cross_encoder = initialize_cross_encoder()
        qa_chain = build_qa_chain(ensemble_retriever, llm)

        print("\nWelcome to the Ensemble Financial Document QA System with Re-ranking!")
        print("Type 'exit' to quit.\n")

        while True:
            query = input("You ➔ ").strip()
            if query.lower() == "exit":
                print("Goodbye!")
                break
            if not query:
                print("Please enter a valid question.\n")
                continue

            # Retrieve initial documents
            retrieved_docs = ensemble_retriever.get_relevant_documents(query)
            if not retrieved_docs:
                print("No relevant documents retrieved.\n")
                continue

            # Deduplicate documents
            unique_docs = deduplicate_documents(retrieved_docs)
            if not unique_docs:
                print("No unique documents after deduplication.\n")
                continue

            # Re-rank documents
            reranked_docs = rerank_documents(query, unique_docs, cross_encoder, top_k=5)
            if not reranked_docs:
                print("No documents after re-ranking.\n")
                continue

            # Create a temporary retriever with re-ranked docs
            try:
                temp_retriever = FAISS.from_documents(
                    documents=reranked_docs,
                    embedding=HuggingFaceEmbeddings(
                        model_name='sentence-transformers/all-MiniLM-L6-v2',
                        model_kwargs={'device': 'cpu'},
                        encode_kwargs={'normalize_embeddings': True}
                    ),
                    ids=[doc.metadata["chunk_id"] for doc in reranked_docs]
                ).as_retriever(search_kwargs={"k": 5})
                qa_chain.retriever = temp_retriever
            except Exception as e:
                print(f"Failed to create FAISS retriever: {e}")
                # Fallback: Construct context manually
                context = "\n".join([doc.page_content for doc in reranked_docs])
                prompt = PromptTemplate(
                    input_variables=["context", "question"],
                    template="""You are a financial expert. Using the following context, answer the question concisely and professionally. If the context is insufficient, provide a best-guess answer using financial reasoning.

Context:
{context}

Question:
{question}

Answer:"""
                )
                response = llm.invoke(prompt.format(context=context, question=query))
                print(f"\nAnswer: {response.content}\n")
                continue

            # Invoke QA chain
            response = qa_chain({"question": query})
            print(f"\nAnswer: {response['answer']}\n")


    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()