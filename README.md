# 📄 AI Document Assistant

A simple Streamlit AI Document Assistant for extracting, chunking, embedding, searching, and asking questions about PDF, DOCX, TXT, and Markdown documents.

## Features

- PDF, DOCX, TXT and MD extraction
- Filename and page metadata
- Overlapping text chunks
- Sentence Transformers embeddings
- FAISS semantic search
- Keyword search
- Hybrid search
- Groq question answering
- Retrieved source display
- Google Drive file links
- Streamlit session state and caching

## Project Structure

```text
ai-document-assistant/
├── app.py
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Groq API Key

Create:

```text
.streamlit/secrets.toml
```

Add:

```toml
GROQ_API_KEY = "your-groq-api-key"
```

Never hardcode the API key in `app.py` or commit `secrets.toml` to GitHub.

## Pipeline

```text
Document
↓
Text Extraction
↓
Chunking
↓
Sentence Transformer Embeddings
↓
FAISS Index
↓
User Question
↓
Semantic + Keyword Search
↓
Hybrid Ranking
↓
Relevant Chunks
↓
Groq
↓
Answer + Sources
```

## Search

The hybrid score combines semantic and keyword scores:

```text
Hybrid Score = 0.7 × Semantic Score + 0.3 × Keyword Score
```

Document embeddings are created when documents are processed and stored in Streamlit session state. They are reused for subsequent questions.

## Google Drive

The simple version supports publicly accessible Google Drive file links. Full folder browsing requires Google Drive API authentication and file listing.
