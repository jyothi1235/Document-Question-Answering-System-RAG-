import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import tempfile
import streamlit as st

from transformers import pipeline
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


st.set_page_config(
    page_title="Document Question Answering System",
    page_icon="📄",
    layout="wide"
)

if st.button("🔄 Clear Previous Data"):
    st.session_state.clear()
    st.cache_resource.clear()
    st.rerun()


st.markdown("""
<style>
.stApp {
    background: linear-gradient(-45deg, #dbeafe, #ede9fe, #fce7f3, #e0f2fe);
    background-size: 400% 400%;
    animation: gradientBG 12s ease infinite;
    color: #111827;
}
@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.main-title {
    text-align: center;
    font-size: 46px;
    font-weight: 800;
    color: #1f2937;
}
.sub-title {
    text-align: center;
    font-size: 18px;
    color: #374151;
    margin-bottom: 30px;
}
.glass-card, .small-box, .answer-box {
    background: rgba(255,255,255,0.80);
    border-radius: 18px;
    padding: 18px;
    margin-bottom: 14px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.08);
}
.answer-box {
    border-left: 8px solid #2563eb;
    font-size: 19px;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)


st.markdown(
    '<div class="main-title">📄 Document Question Answering System</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="sub-title">Upload PDF, TXT, DOCX, or CSV documents and ask questions using RAG</div>',
    unsafe_allow_html=True
)


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def load_qa_pipeline():
    return pipeline(
        "question-answering",
        model="deepset/roberta-base-squad2"
    )


def get_loader(file_path, file_name):
    extension = file_name.split(".")[-1].lower()

    if extension == "pdf":
        return PyPDFLoader(file_path)

    if extension == "txt":
        return TextLoader(file_path, encoding="utf-8")

    if extension == "csv":
        return CSVLoader(file_path)

    if extension == "docx":
        return Docx2txtLoader(file_path)

    return None


def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=200
    )
    return splitter.split_documents(docs)


def generate_answer(query, retrieved_docs, qa_model):
    context = "\n\n".join(doc.page_content for doc in retrieved_docs).strip()

    if not context:
        return "", 0.0

    result = qa_model(
        question=query,
        context=context
    )

    return result.get("answer", "").strip(), result.get("score", 0.0)


def is_good_answer(answer, score):
    if not answer:
        return False

    bad_answers = {"[cls]", "unknown", "empty", ".", ".."}

    if answer.strip().lower() in bad_answers:
        return False

    if len(answer.strip()) < 2:
        return False

    if score < 0.01:
        return False

    return True


uploaded_files = st.file_uploader(
    "📤 Upload Documents",
    type=["pdf", "txt", "docx", "csv"],
    accept_multiple_files=True
)

MAX_SIZE_MB = 5

if uploaded_files:
    for file in uploaded_files:
        file_size_mb = file.size / (1024 * 1024)

        if file_size_mb > MAX_SIZE_MB:
            st.error(
                f"❌ {file.name} is {file_size_mb:.2f} MB. "
                f"Please upload below {MAX_SIZE_MB} MB for smooth demo."
            )
            st.stop()


if uploaded_files:
    embeddings = load_embeddings()
    qa_model = load_qa_pipeline()

    all_documents = []

    with st.spinner("Processing uploaded documents..."):
        for idx, file in enumerate(uploaded_files, start=1):
            extension = file.name.split(".")[-1].lower()

            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{extension}") as tmp:
                tmp.write(file.getvalue())
                temp_path = tmp.name

            loader = get_loader(temp_path, file.name)

            if loader is None:
                st.error(f"Unsupported file type: {file.name}")
                os.remove(temp_path)
                continue

            docs = loader.load()
            os.remove(temp_path)

            for doc in docs:
                doc.metadata["source"] = file.name
                doc.metadata["document_number"] = f"Document {idx}"
                doc.metadata["file_type"] = extension.upper()

            all_documents.extend(docs)

        chunks = split_docs(all_documents)
        vectorstore = FAISS.from_documents(chunks, embeddings)

    st.markdown(
        f"""
        <div class="glass-card">
            <h3>✅ Documents loaded successfully</h3>
            <p><b>Files uploaded:</b> {len(uploaded_files)}</p>
            <p><b>Total document sections:</b> {len(all_documents)}</p>
            <p><b>Total chunks:</b> {len(chunks)}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 📂 Uploaded Documents")

    for i, file in enumerate(uploaded_files, start=1):
        st.markdown(
            f"""
            <div class="small-box">
                <b>Document {i}:</b> {file.name}
            </div>
            """,
            unsafe_allow_html=True
        )

    query = st.text_input("💬 Ask a question from the documents")

    if query:
        with st.spinner("Searching and generating answer..."):
            docs_scores = vectorstore.similarity_search_with_score(query, k=5)
            retrieved_docs = [doc for doc, score in docs_scores]

            answer, qa_score = generate_answer(query, retrieved_docs, qa_model)

        st.markdown("## 📌 Answer")

        top_doc = retrieved_docs[0] if retrieved_docs else None

        if top_doc:
            source = top_doc.metadata.get("source", "Unknown")
            doc_number = top_doc.metadata.get("document_number", "Document")
            file_type = top_doc.metadata.get("file_type", "N/A")
            page = top_doc.metadata.get("page", "N/A")
            page_display = page + 1 if isinstance(page, int) else page

            st.markdown(
                f"""
                <div class="small-box">
                    <b>Matched Document:</b> {doc_number}<br>
                    <b>File Name:</b> {source}<br>
                    <b>File Type:</b> {file_type}<br>
                    <b>Page/Section:</b> {page_display}
                </div>
                """,
                unsafe_allow_html=True
            )

        if is_good_answer(answer, qa_score):
            st.markdown(
                f"""
                <div class="answer-box">
                    <b>Answer:</b><br><br>
                    {answer}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="small-box">
                    <b>Confidence Score:</b> {qa_score:.4f}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            if retrieved_docs:
                st.markdown(
                    f"""
                    <div class="answer-box">
                        <b>Possible Answer:</b><br><br>
                        {retrieved_docs[0].page_content[:700]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.info("Exact answer was not clearly extracted, so the most relevant content is shown.")
            else:
                st.warning("No relevant content found in uploaded documents.")

        st.markdown("## 🔎 Retrieved Chunks with Similarity Score")

        for i, (doc, score) in enumerate(docs_scores, start=1):
            source = doc.metadata.get("source", "Unknown")
            doc_number = doc.metadata.get("document_number", "Document")
            file_type = doc.metadata.get("file_type", "N/A")
            page = doc.metadata.get("page", "N/A")
            page_display = page + 1 if isinstance(page, int) else page

            with st.expander(
                f"Chunk {i} | {doc_number} | {source} | {file_type} | Page/Section {page_display}"
            ):
                st.write(doc.page_content)
                st.caption(f"Similarity Score: {score:.4f}")

else:
    st.markdown(
        """
        <div class="glass-card">
            <h3>Upload documents to begin</h3>
            <p>Supported formats: PDF, TXT, DOCX, CSV.</p>
            <p>After uploading, ask a question and the system will retrieve relevant content and generate an answer.</p>
        </div>
        """,
        unsafe_allow_html=True
    )