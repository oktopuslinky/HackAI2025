import pymupdf as fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    all_text = ""

    for page_num in range(len(doc)):
        page = doc.load_page(page_num) 
        text = page.get_text()
        all_text += f"\n--- Page {page_num + 1} ---\n{text}"

    return all_text

def chunk_raw_text(pdf_path, chunk_size=500, chunk_overlap=50):
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

pdf_path = "ltimindtree_annual_report.pdf"
extracted_text = extract_text_from_pdf(pdf_path)
documents = chunk_raw_text(pdf_path)
print(documents[20])
embeddings = initialize_embeddings()
#print(embeddings)
vector_store = create_vector_store(documents, embeddings)
#print(vector_store)
query = "What was LTIMindTree's revenue growth in 2024?"

relevant_docs = dense_retrieval(query, vector_store, k=5)
#print(relevant_docs)
'''for i, doc in enumerate(relevant_docs):
    print(f"\nDocument {i}:")
    print(f"Content: {doc.page_content[:200]}...")  # Truncate for brevity
    print(f"Metadata: {doc.metadata}")'''
    



# Print or save
#print(extracted_text)
