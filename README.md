# Multi-PDF-RAG-App-with-Sources

A Streamlit + LangChain app that uploads multiple PDFs, indexes them into a single FAISS vector store, and answers questions with source file and page references.

## Setup

1. Create/activate a virtual environment.
2. Install dependencies (the Hugging Face embedding model is downloaded on first use):
   ```bash
   pip install -r requirements.txt
   ```
3. Set your OpenAI API key (used for answer generation; PDF embeddings run locally with Hugging Face):
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   ```
4. Run the app:
   ```bash
   streamlit run rag_app.py
   ```

## Features

- Upload multiple PDF files at once
- Chunk and embed all uploaded PDFs into one FAISS index
- Generate local embeddings with `sentence-transformers/all-MiniLM-L6-v2`
- Store the vector DB in `st.session_state`
- Answer questions using retrieval across the uploaded documents
- Show source filename and page number for every result
