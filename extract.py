import pymupdf as fitz  # PyMuPDF
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import os
def initialize_llm():      # LLM initializing
    try:
        llm = ChatGroq(
            model_name="llama3-70b-8192",  
            groq_api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.5,
            max_tokens=500
        )
        print("Groq LLM initialized")
        return llm
    except Exception as e:
        raise Exception(f"Failed to initialize Groq LLM: {str(e)}")

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    all_text = ""

    for page_num in range(len(doc)):
        page = doc.load_page(page_num) 
        text = page.get_text()
        all_text += f"\n--- Page {page_num + 1} ---\n{text}"
        all_text = ' '.join(all_text.split())
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
    try:
        if not query.strip():
            raise ValueError("Query is empty.")
        if not hasattr(vector_store, 'index') or vector_store.index.ntotal == 0:
            raise ValueError("Vector store is empty or invalid.")
        retriever = vector_store.as_retriever(search_kwargs={"k": k})
        print(f"Retriever created: {retriever}")
        
      
        query_embedding = vector_store.embedding_function.embed_query(query)
        print(f"Query embedding length: {len(query_embedding)}")
        
    
        doc_score_pairs = vector_store.similarity_search_with_score(query, k=k)
        print(f"Retrieved {len(doc_score_pairs)} documents with scores")
        
        # Extract documents and scores
        relevant_docs = [pair[0] for pair in doc_score_pairs]
        scores = [pair[1] for pair in doc_score_pairs]
        
        
        if not relevant_docs:
            print("No documents retrieved for query.")
        return relevant_docs,scores
    except Exception as e:
        raise Exception(f"Failed to perform dense retrieval: {str(e)}")

def generate_answer(llm, query, relevant_docs, scores):
    try:
        
        context = "\n".join([f"Document {i+1} (Score: {score:.3f}):\n{doc.page_content}" 
                             for i, (doc, score) in enumerate(zip(relevant_docs, scores))])
        
        
        prompt_template = PromptTemplate(
            input_variables=["query", "context"],
            template="""You are a financial analyst. Answer the query concisely using the provided context from LTIMindTree's annual report. If the context lacks sufficient information, say so and provide a general answer based on available data.

Query: {query}

Context:
{context}

Answer:"""
        )
        
        # generating response by giving query and context
        chain = prompt_template | llm
        response = chain.invoke({"query": query, "context": context})
        return response.content
    except Exception as e:
        raise Exception(f"Failed to generate answer: {str(e)}")
    
pdf_path = "ltimindtree_annual_report.pdf"
extracted_text = extract_text_from_pdf(pdf_path)
#print(extracted_text)
documents = chunk_raw_text(pdf_path)
#print(documents[20])
embeddings = initialize_embeddings()

vector_store = create_vector_store(documents, embeddings)

query = "What are the activities to sustain value?"

relevant_docs,scores = dense_retrieval(query, vector_store, k=5)
print(relevant_docs[0])
#print(relevant_docs)

llm = initialize_llm()
answer = generate_answer(llm, query, relevant_docs, scores)
print(answer)     # checking if llm answer is good


#print(extracted_text)
