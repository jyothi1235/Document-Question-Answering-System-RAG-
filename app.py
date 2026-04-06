import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline


# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Document Question Answering System",
    page_icon="📄",
    layout="wide"
)


# -------------------------------------------------
# Custom CSS
# -------------------------------------------------
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

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

.main-title {
    text-align: center;
    font-size: 48px;
    font-weight: 800;
    color: #1f2937;
    margin-bottom: 10px;
}

.sub-title {
    text-align: center;
    font-size: 18px;
    color: #374151;
    margin-bottom: 30px;
}

.glass-card {
    background: rgba(255, 255, 255, 0.70);
    backdrop-filter: blur(12px);
    border-radius: 22px;
    padding: 20px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.08);
    border: 1px solid rgba(255,255,255,0.45);
    margin-bottom: 20px;
}

.answer-box {
    background: linear-gradient(135deg, #ffffff, #eff6ff);
    color: #111827;
    border-left: 8px solid #2563eb;
    border-radius: 18px;
    padding: 22px;
    font-size: 20px;
    font-weight: 500;
    box-shadow: 0 6px 18px rgba(37, 99, 235, 0.12);
    margin-top: 10px;
}

.small-box {
    background: rgba(255,255,255,0.75);
    border-radius: 16px;
    padding: 14px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.06);
    margin-bottom: 12px;
}

.chunk-box {
    background: rgba(255,255,255,0.82);
    border-radius: 14px;
    padding: 14px;
    margin-bottom: 10px;
    border-left: 6px solid #7c3aed;
}

.stTextInput > div > div > input {
    background: rgba(255,255,255,0.92);
    border-radius: 14px;
    border: 1px solid #cbd5e1;
    color: #111827;
    font-size: 18px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------
# Header
# -------------------------------------------------
st.markdown('<div class="main-title">📄 Document Question Answering System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload PDF documents and ask questions about them.</div>', unsafe_allow_html=True)


# -------------------------------------------------
# File Uploader
# -------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload PDF Documents",
    type="pdf",
    accept_multiple_files=True
)


# -------------------------------------------------
# Load Shared Resources
# -------------------------------------------------
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


def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=150
    )
    return splitter.split_documents(docs)


def generate_answer(query, retrieved_docs, qa_model):
    context = "\n\n".join(doc.page_content for doc in retrieved_docs).strip()

    if not context:
        return "", 0.0

    result = qa_model(question=query, context=context)
    answer = result.get("answer", "").strip()
    score = result.get("score", 0.0)
    return answer, score


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


def filename_bonus(query, filename):
    query_lower = query.lower()
    filename_lower = filename.lower()
    bonus = 0.0

    for word in query_lower.split():
        if len(word) > 3 and word in filename_lower:
            bonus += 0.15

    return bonus


# -------------------------------------------------
# Main
# -------------------------------------------------
if uploaded_files:
    embeddings = load_embeddings()
    qa_model = load_qa_pipeline()

    pdf_stores = []
    total_pages = 0
    total_chunks = 0

    for idx, file in enumerate(uploaded_files, start=1):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            temp_path = tmp.name

        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        total_pages += len(docs)

        for d in docs:
            d.metadata["source"] = file.name
            d.metadata["pdf_number"] = f"PDF {idx}"

        chunks = split_docs(docs)
        total_chunks += len(chunks)

        vectorstore = FAISS.from_documents(chunks, embeddings)

        pdf_stores.append({
            "pdf_number": f"PDF {idx}",
            "file_name": file.name,
            "docs": docs,
            "chunks": chunks,
            "vectorstore": vectorstore
        })

    st.markdown(
        f"""
        <div class="glass-card">
            <h3 style="margin-top:0;">✅ Documents loaded successfully</h3>
            <p><b>Files uploaded:</b> {len(uploaded_files)}</p>
            <p><b>Total pages loaded:</b> {total_pages}</p>
            <p><b>Total chunks created:</b> {total_chunks}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Uploaded PDFs")
    for pdf in pdf_stores:
        st.markdown(
            f"""
            <div class="small-box">
                <b>{pdf["pdf_number"]}:</b> {pdf["file_name"]}
            </div>
            """,
            unsafe_allow_html=True
        )

    query = st.text_input("Ask a question from the documents")

    if query:
        with st.spinner("Searching and generating answer..."):
            best_pdf = None
            best_adjusted_score = float("inf")
            best_docs_scores = None

            for pdf in pdf_stores:
                docs_scores = pdf["vectorstore"].similarity_search_with_score(query, k=3)

                if docs_scores:
                    top_score = docs_scores[0][1]
                    adjusted_score = top_score - filename_bonus(query, pdf["file_name"])

                    if adjusted_score < best_adjusted_score:
                        best_adjusted_score = adjusted_score
                        best_pdf = pdf
                        best_docs_scores = docs_scores

            if best_pdf and best_docs_scores:
                retrieved_docs = [doc for doc, score in best_docs_scores]
                answer, qa_score = generate_answer(query, retrieved_docs, qa_model)
            else:
                retrieved_docs = []
                answer, qa_score = "", 0.0

        st.markdown("## Answer")

        if not best_pdf or not best_docs_scores:
            st.warning("No relevant content was found in the uploaded PDF documents.")
        else:
            top_doc = best_docs_scores[0][0]
            top_source = top_doc.metadata.get("source", "Unknown")
            top_pdf_number = top_doc.metadata.get("pdf_number", "PDF")
            top_page = top_doc.metadata.get("page", "N/A")
            top_page_display = top_page + 1 if isinstance(top_page, int) else top_page

            st.markdown(
                f"""
                <div class="small-box">
                    <b>Matched PDF:</b> {top_pdf_number}<br>
                    <b>File Name:</b> {top_source}<br>
                    <b>Page:</b> {top_page_display}
                </div>
                """,
                unsafe_allow_html=True
            )

            if is_good_answer(answer, qa_score):
                st.markdown(
                    f"""
                    <div class="answer-box">
                        <b>Answer extracted from {top_pdf_number}:</b><br><br>
                        {answer}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"""
                    <div class="small-box">
                        <b>Confidence score:</b> {qa_score:.4f}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                fallback_text = retrieved_docs[0].page_content[:500] if retrieved_docs else ""
                if fallback_text:
                    st.markdown(
                        f"""
                        <div class="answer-box">
                            <b>Possible answer from {top_pdf_number}:</b><br>
                            <b>File:</b> {top_source}<br>
                            <b>Page:</b> {top_page_display}<br><br>
                            {fallback_text}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.info("Exact answer was not extracted clearly, so the most relevant content from the best matched PDF is shown.")
                else:
                    st.warning("This question is not clearly answered in the uploaded PDF documents.")

            st.markdown("## Retrieved Chunks with Similarity Score")
            for i, (doc, score) in enumerate(best_docs_scores, start=1):
                source = doc.metadata.get("source", "Unknown")
                pdf_number = doc.metadata.get("pdf_number", "PDF")
                page = doc.metadata.get("page", "N/A")
                page_display = page + 1 if isinstance(page, int) else page

                st.markdown(
                    f"""
                    <div class="chunk-box">
                        <b>Chunk {i}</b><br>
                        <b>PDF:</b> {pdf_number}<br>
                        <b>File:</b> {source}<br>
                        <b>Page:</b> {page_display}<br>
                        <b>Similarity Score:</b> {score:.4f}<br><br>
                        {doc.page_content}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("## Source Information")
            grouped_sources = {}
            for doc, score in best_docs_scores:
                source = doc.metadata.get("source", "Unknown")
                pdf_number = doc.metadata.get("pdf_number", "PDF")
                page = doc.metadata.get("page", "N/A")
                page_display = page + 1 if isinstance(page, int) else page
                key = f"{pdf_number} | {source} | Page {page_display}"

                if key not in grouped_sources:
                    grouped_sources[key] = []
                grouped_sources[key].append((doc.page_content, score))

            for source_key, items in grouped_sources.items():
                with st.expander(f"📄 {source_key}"):
                    for idx, (text, score) in enumerate(items, start=1):
                        st.write(f"**Chunk {idx} Similarity Score:** {score:.4f}")
                        st.write(text)
                        st.write("---")

else:
    st.markdown(
        """
        <div class="glass-card">
            <h3 style="margin-top:0;">Upload one or more PDF documents to begin</h3>
            <p>After uploading, type a question and the system will retrieve the best matching PDF, show chunks, similarity scores, and generate an answer.</p>
        </div>
        """,
        unsafe_allow_html=True
    )