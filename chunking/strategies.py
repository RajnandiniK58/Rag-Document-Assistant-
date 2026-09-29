import re

def fixed_size_chunk(text, chunk_size=500, overlap=50):
    """Split text into fixed-size character chunks with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap  # overlap keeps context from being cut off harshly
    return chunks

def sentence_chunk(text, sentences_per_chunk=5):
    """Split text by sentences, grouping N sentences per chunk."""
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(sentences[i:i+sentences_per_chunk])
        chunks.append(chunk)
    return chunks

def paragraph_chunk(text):
    """Split text by paragraph breaks (double newlines)."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paragraphs

def semantic_chunk(text, model, similarity_threshold=0.5):
    """Group sentences into chunks based on semantic similarity."""
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    sentences = re.split(r'(?<=[.!?]) +', text)
    sentences = [s for s in sentences if s.strip()]
    if len(sentences) == 0:
        return []

    embeddings = model.encode(sentences)
    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = cosine_similarity([embeddings[i-1]], [embeddings[i]])[0][0]
        if sim >= similarity_threshold:
            current_chunk.append(sentences[i])
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
    chunks.append(" ".join(current_chunk))
    return chunks