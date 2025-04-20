##Inspiration
Have you ever tried to summarize a large, complex financial document with traditional AI and been unsatisfied with the results? Maybe it doesn't cover enough, or it hallucinates. Well, we have a solution for you!

##What it does
RAGs2Riches is a MultiModal RAG application which allows users to upload any file they have especially financial reports, and receive answers to complex queries through our LLM. Simply upload your file, start asking questions, and let the magic flow!

##How we built it
RAGs2Riches isn't your typical RAG app. We first extract all the text from the uploaded pdf. We then send it over to two separate retrieval functions, namely Dense Retrieval (using vector similarity in FAISS), and Sparse Retrieval (using keyword-based matching), and then put those 2 together into one Ensemble Retrieval which combines the strength of both, complimenting each other to be an even more accurate retrieval technique. It doesn't just stop there. We even have re-ranking after the ensemble model, to sort and search the retrieved documents to reduce hallucination and ensure an accurate response from the LLM(which is Llama).

##Challenges we ran into
Re-Ranking proved to be a challenge because when we get docs from both retrievers, some of them had overlapping IDs, so when we tried creating a temporary retriever from FAISS for re-ranking, there was an error due to the duplicate IDs.

##How to run locally
1. Clone the repository
git clone https://github.com/your-username/rags2riches.git
cd rags2riches

2. Run the server using uvicorn main:app --reload

3. Set up frontend (React + Vite + ShadCN)

cd frontend

# Install frontend dependencies
npm install

# Run the Vite dev server
npm run dev

4. Create a .env file in the backend root
GROQ_API_KEY=your_groq_api_key_here



