# RAG-Powered Document Intelligence Assistant

An AI assistant that answers questions grounded in your own documents. Upload a PDF, DOCX, TXT, or CSV file, choose a chunking strategy, and ask natural-language questions — the app retrieves the most relevant sections of your document and uses Google Gemini to generate an answer based only on that content, with source chunks shown for transparency.

## How it works

1. **Upload & extract** — the document is parsed based on file type (PDF via `pypdf`, DOCX via `python-docx`, CSV via `pandas`, or plain text).
2. **Chunk** — the extracted text is split into chunks using one of four selectable strategies (see below).
3. **Embed & index** — each chunk is embedded using `all-MiniLM-L6-v2` (SentenceTransformers) and indexed with FAISS (`IndexFlatL2`) for similarity search.
4. **Retrieve** — when a question is asked, it's embedded the same way, and the top-k most similar chunks are retrieved via L2 distance.
5. **Generate** — the retrieved chunks are passed as context to Gemini 2.5 Flash in a grounded prompt, so the model only answers from what's actually in the document (and says so if the answer isn't there).
6. **Display** — the answer is shown along with an expandable view of the exact source chunks used, with their similarity distance scores.

## Chunking strategies

- **Fixed-size** — splits text into fixed-length character chunks (default 500 chars) with overlap (default 50 chars) so context isn't harshly cut off at chunk boundaries.
- **Sentence** — groups a fixed number of sentences (default 5) per chunk, using regex-based sentence splitting.
- **Paragraph** — splits on double newlines, treating each paragraph as its own chunk.
- **Semantic** — embeds each sentence individually and greedily groups consecutive sentences into a chunk while their cosine similarity stays above a threshold (default 0.5), starting a new chunk when the topic shifts.

The strategy is user-selectable in the UI so results can be compared directly; in a production setting this would typically default automatically based on document type/structure rather than requiring the user to choose.

## Tech stack

- **Frontend:** Streamlit
- **Embeddings:** SentenceTransformers (`all-MiniLM-L6-v2`)
- **Vector search:** FAISS (`IndexFlatL2`)
- **LLM:** Google Gemini 2.5 Flash (`google-generativeai`)
- **Document parsing:** `pypdf`, `python-docx`, `pandas`

## Project structure

```
app.py                     # Streamlit app: UI, document loading, embedding, retrieval, generation
chunking/
  strategies.py             # Fixed-size, sentence, paragraph, and semantic chunking functions
requirements.txt
```

## Setup

1. Clone the repo and create a virtual environment:
   ```
   git clone https://github.com/RajnandiniK58/Rag-Document-Assistant-.git
   cd Rag-Document-Assistant-
   python -m venv venv
   venv\Scripts\activate   # Windows
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root with your Gemini API key:
   ```
   GOOGLE_API_KEY=your_key_here
   ```

## Running locally

```
streamlit run app.py
```

Open the URL Streamlit prints (typically `http://localhost:8501`), upload a document, pick a chunking strategy, and ask a question.

## Notes

- The app uses `st.session_state` to persist the built index and chunks across interactions within a session.
- Retrieval uses simple top-k nearest neighbor search (k=8) via FAISS's flat L2 index — exact search, not approximate, which is fine at the scale of a single uploaded document but wouldn't scale to a large multi-document corpus without a different index type (e.g. IVF or HNSW).
