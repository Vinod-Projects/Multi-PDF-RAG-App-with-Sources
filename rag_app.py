import os
import tempfile

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="Multi-PDF RAG App", page_icon="📚")
st.title("Multi-PDF RAG App with Sources")


def index_uploaded_pdfs(uploaded_files):
    if not uploaded_files:
        st.warning("Upload one or more PDF files to build the vector store.")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(uploaded_file.getvalue())
            temp_path = temp_pdf.name

        try:
            loader = PyPDFLoader(temp_path)
            pages = loader.load()
            for document in pages:
                document.metadata["source"] = uploaded_file.name
                document.metadata["page"] = document.metadata.get("page", 1)
            chunks.extend(splitter.split_documents(pages))
        finally:
            try:
                os.remove(temp_path)
            except FileNotFoundError:
                pass

    if not chunks:
        st.error("No readable PDF pages were found in the uploaded files.")
        return

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_documents(chunks, embedding_model)
    st.session_state.store = vector_store
    st.session_state.chunk_count = len(chunks)
    st.success(f"Indexed {len(chunks)} chunks across {len(uploaded_files)} PDF(s).")


def answer_question(question):
    if not os.getenv("OPENAI_API_KEY"):
        st.error("Missing OPENAI_API_KEY environment variable. It is required to generate answers.")
        return

    store = st.session_state.get("store")
    if store is None:
        st.warning("Please upload and index PDFs before asking a question.")
        return

    docs = store.similarity_search(question, k=4)
    if not docs:
        st.warning("No relevant passages were found in the uploaded documents.")
        return

    context = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')} | Page: {doc.metadata.get('page', 'unknown')}\n{doc.page_content}"
        for doc in docs
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer the user's question using only the supplied context. "
                "If the sources disagree, say so clearly and explain the conflict. "
                "Whenever possible, cite the source file and page number."
            ),
            ("user", "Question: {question}\n\nContext:\n{context}"),
        ]
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    answer_chain = prompt | llm | StrOutputParser()
    answer = answer_chain.invoke({"question": question, "context": context})

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources")
    for doc in docs:
        source_name = doc.metadata.get("source", "unknown")
        page_number = doc.metadata.get("page", "unknown")
        st.caption(f"{source_name} — page {page_number}")


with st.sidebar:
    st.header("Upload PDFs")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
    )
    if st.button("Build / Rebuild Index"):
        index_uploaded_pdfs(uploaded_files)

    if "chunk_count" in st.session_state:
        st.caption(f"Indexed chunks: {st.session_state.chunk_count}")


if "store" in st.session_state:
    with st.form("question_form"):
        question = st.text_input("Ask a question about the uploaded PDFs")
        submitted = st.form_submit_button("Ask")

    if submitted and question:
        answer_question(question)
else:
    st.info("Upload one or more PDFs and click 'Build / Rebuild Index' to create the search index.")
