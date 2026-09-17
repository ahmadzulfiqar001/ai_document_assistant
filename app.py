import io
import re
import requests
import streamlit as st
import numpy as np
import faiss

from pypdf import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
from groq import Groq

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Document Assistant")
st.write("Upload documents or load them from Google Drive, then ask questions about their content.")


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def get_groq_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])


def extract_pdf(file_bytes, filename):
    documents = []
    reader = PdfReader(io.BytesIO(file_bytes))

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            documents.append({
                "filename": filename,
                "page": page_number,
                "text": text.strip()
            })

    return documents


def extract_docx(file_bytes, filename):
    document = Document(io.BytesIO(file_bytes))

    text = "\n".join(
        paragraph.text
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    )

    if not text.strip():
        return []

    return [{
        "filename": filename,
        "page": None,
        "text": text.strip()
    }]


def extract_txt(file_bytes, filename):
    text = file_bytes.decode("utf-8", errors="ignore")

    if not text.strip():
        return []

    return [{
        "filename": filename,
        "page": None,
        "text": text.strip()
    }]


def extract_md(file_bytes, filename):
    text = file_bytes.decode("utf-8", errors="ignore")

    if not text.strip():
        return []

    return [{
        "filename": filename,
        "page": None,
        "text": text.strip()
    }]


def extract_document(file_bytes, filename):
    extension = filename.lower().split(".")[-1]

    if extension == "pdf":
        return extract_pdf(file_bytes, filename)

    if extension == "docx":
        return extract_docx(file_bytes, filename)

    if extension == "txt":
        return extract_txt(file_bytes, filename)

    if extension == "md":
        return extract_md(file_bytes, filename)

    return []


def create_chunks(documents, chunk_size=800, overlap=150):
    chunks = []

    for document in documents:
        text = document["text"]
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "chunk_id": len(chunks),
                    "filename": document["filename"],
                    "page": document["page"],
                    "text": chunk_text
                })

            if end >= len(text):
                break

            start += chunk_size - overlap

    return chunks


def create_embeddings(chunks):
    model = load_embedding_model()
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embeddings.astype("float32")


def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


def important_words(question):
    words = re.findall(r"\b[a-zA-Z0-9]+\b", question.lower())

    stop_words = {
        "what", "when", "where", "who", "why", "how",
        "is", "are", "was", "were", "the", "a", "an",
        "of", "to", "in", "on", "for", "and", "or",
        "can", "does", "do", "did", "from", "with"
    }

    return [
        word for word in words
        if word not in stop_words and len(word) > 2
    ]


def keyword_search(question, chunks):
    words = important_words(question)
    results = []

    for chunk in chunks:
        text = chunk["text"].lower()

        matches = sum(
            1 for word in words
            if word in text
        )

        score = matches / max(len(words), 1)

        results.append({
            "chunk": chunk,
            "keyword_score": score
        })

    return results


def hybrid_search(question, chunks, index, top_k=5):
    model = load_embedding_model()

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    semantic_scores, semantic_ids = index.search(
        question_embedding,
        len(chunks)
    )

    semantic_score_map = {}

    for score, chunk_id in zip(
        semantic_scores[0],
        semantic_ids[0]
    ):
        if chunk_id >= 0:
            semantic_score_map[int(chunk_id)] = float(score)

    keyword_results = keyword_search(question, chunks)
    results = []

    for item in keyword_results:
        chunk = item["chunk"]
        chunk_id = chunk["chunk_id"]

        semantic_score = semantic_score_map.get(chunk_id, 0.0)
        keyword_score = item["keyword_score"]

        hybrid_score = (
            0.7 * semantic_score +
            0.3 * keyword_score
        )

        results.append({
            "chunk": chunk,
            "semantic_score": semantic_score,
            "keyword_score": keyword_score,
            "hybrid_score": hybrid_score
        })

    results.sort(
        key=lambda x: x["hybrid_score"],
        reverse=True
    )

    return results[:top_k]


def answer_question(question, retrieved_chunks):
    client = get_groq_client()
    context_parts = []

    for item in retrieved_chunks:
        chunk = item["chunk"]

        page_text = (
            f"Page {chunk['page']}"
            if chunk["page"] is not None
            else "Page not available"
        )

        context_parts.append(
            f"File: {chunk['filename']}\n"
            f"{page_text}\n"
            f"Content:\n{chunk['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""
You are an AI Document Assistant.

Answer the user's question using ONLY the provided document context.

If the answer is not available in the context, clearly say:
"I could not find this information in the provided documents."

Do not use outside knowledge.
Do not invent facts.
Keep the answer clear and concise.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": "Answer only from the provided document context."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content


def get_drive_file_id(url):
    match = re.search(
        r"/file/d/([a-zA-Z0-9_-]+)",
        url
    )

    if match:
        return match.group(1)

    match = re.search(
        r"id=([a-zA-Z0-9_-]+)",
        url
    )

    if match:
        return match.group(1)

    return None


def download_drive_file(url):
    file_id = get_drive_file_id(url)

    if not file_id:
        return None, None

    download_url = (
        f"https://drive.google.com/uc"
        f"?export=download&id={file_id}"
    )

    response = requests.get(
        download_url,
        timeout=30
    )

    if response.status_code != 200:
        return None, None

    content_type = response.headers.get(
        "content-type",
        ""
    )

    filename = f"drive_file_{file_id}"

    if "pdf" in content_type:
        filename += ".pdf"
    elif "word" in content_type:
        filename += ".docx"
    elif "text" in content_type:
        filename += ".txt"
    else:
        return None, None

    return response.content, filename


if "documents" not in st.session_state:
    st.session_state.documents = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

if "processed" not in st.session_state:
    st.session_state.processed = False


st.sidebar.header("📂 Document Sources")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF, DOCX, TXT or MD",
    type=["pdf", "docx", "txt", "md"],
    accept_multiple_files=True
)

st.sidebar.write("Or load a Google Drive file:")

drive_url = st.sidebar.text_input(
    "Google Drive file link"
)

process_button = st.sidebar.button(
    "Process Documents"
)


if process_button:
    documents = []

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_bytes = uploaded_file.getvalue()

            extracted = extract_document(
                file_bytes,
                uploaded_file.name
            )

            documents.extend(extracted)

    if drive_url.strip():
        file_bytes, filename = download_drive_file(
            drive_url.strip()
        )

        if file_bytes and filename:
            extracted = extract_document(
                file_bytes,
                filename
            )

            documents.extend(extracted)
        else:
            st.sidebar.error(
                "Could not load the Google Drive file. "
                "Make sure the file is publicly accessible."
            )

    if documents:
        chunks = create_chunks(documents)
        embeddings = create_embeddings(chunks)
        faiss_index = create_faiss_index(embeddings)

        st.session_state.documents = documents
        st.session_state.chunks = chunks
        st.session_state.embeddings = embeddings
        st.session_state.faiss_index = faiss_index
        st.session_state.processed = True

        st.sidebar.success(
            f"Created {len(chunks)} chunks."
        )
    else:
        st.sidebar.warning(
            "No supported documents were loaded."
        )


if st.session_state.processed:
    documents = st.session_state.documents
    chunks = st.session_state.chunks

    st.header("📊 Document Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Documents",
            len(set(
                document["filename"]
                for document in documents
            ))
        )

    with col2:
        st.metric(
            "Chunks",
            len(chunks)
        )

    with col3:
        st.metric(
            "Embedding Size",
            st.session_state.embeddings.shape[1]
        )

    st.divider()

    st.header("📑 Extracted Text")

    for document in documents:
        location = (
            f"Page {document['page']}"
            if document["page"] is not None
            else "Page not available"
        )

        with st.expander(
            f"{document['filename']} — {location}"
        ):
            st.write(document["text"])

    st.divider()

    st.header("✂️ Document Chunks")

    st.write(
        f"Total chunks created: **{len(chunks)}**"
    )

    for chunk in chunks[:10]:
        location = (
            f"Page {chunk['page']}"
            if chunk["page"] is not None
            else "Page not available"
        )

        with st.expander(
            f"Chunk {chunk['chunk_id']} — "
            f"{chunk['filename']} — {location}"
        ):
            st.write(chunk["text"])

    st.divider()

    st.header("💬 Ask Your Documents")

    question = st.text_input(
        "Enter your question"
    )

    top_k = st.slider(
        "Number of sources to retrieve",
        min_value=1,
        max_value=min(10, len(chunks)),
        value=min(5, len(chunks))
    )

    if st.button("Ask Question") and question.strip():
        with st.spinner("Searching documents..."):
            retrieved = hybrid_search(
                question,
                chunks,
                st.session_state.faiss_index,
                top_k
            )

            answer = answer_question(
                question,
                retrieved
            )

        st.subheader("🤖 Answer")
        st.write(answer)

        st.subheader("📚 Retrieved Sources")

        for number, item in enumerate(
            retrieved,
            start=1
        ):
            chunk = item["chunk"]

            location = (
                f"Page {chunk['page']}"
                if chunk["page"] is not None
                else "Page not available"
            )

            with st.expander(
                f"Source {number}: "
                f"{chunk['filename']} — {location}"
            ):
                st.write(chunk["text"])

                st.caption(
                    f"Semantic score: "
                    f"{item['semantic_score']:.3f} | "
                    f"Keyword score: "
                    f"{item['keyword_score']:.3f} | "
                    f"Hybrid score: "
                    f"{item['hybrid_score']:.3f}"
                )
else:
    st.info(
        "Upload a document or provide a Google Drive link, "
        "then click **Process Documents**."
    )
