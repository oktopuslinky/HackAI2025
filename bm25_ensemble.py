from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import pipeline

def build_ensemble_retriever(text_file_path: str):
    loader = TextLoader(text_file_path)
    documents = loader.load()
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Larger chunks for better context
        chunk_overlap=200,
    )
    splits = text_splitter.split_documents(documents)
    
    # Create multiple BM25 retrievers with different parameters
    bm25_retriever1 = BM25Retriever.from_documents(splits)
    bm25_retriever1.k = 3  # Get top 3 results
    
    bm25_retriever2 = BM25Retriever.from_documents(splits)
    bm25_retriever2.k = 5  # Get top 5 results
    
    # Create ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever1, bm25_retriever2],
        weights=[0.3, 0.7]  # Equal weights for both retrievers
    )
    
    return ensemble_retriever

def build_qa_chain(retriever):
    # Create prompt template
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""Answer the question based on the context below. If the answer cannot be found in the context, say "I don't know."

Context: {context}
Question: {question}
Answer:"""
    )
    
    # Create HuggingFace LLM
    hf_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_length=256,
        temperature=0.3,
        do_sample=False,
        device=-1  # CPU
    )
    
    local_llm = HuggingFacePipeline(pipeline=hf_pipeline)
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=local_llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain

if __name__ == "__main__":
    # Initialize the ensemble retriever
    text_file_path = "extracted_text.txt"
    retriever = build_ensemble_retriever(text_file_path)
    
    # Build QA chain
    qa_chain = build_qa_chain(retriever)
    
    print("\nWelcome to the Financial Document QA System!")
    print("Type 'exit' to quit\n")
    
    while True:
        query = input("Your question: ")
        if query.lower() == "exit":
            print("Goodbye!")
            break
            
        # Get answer
        response = qa_chain.invoke({"query": query})
        print(f"\nAnswer: {response['result']}\n") 