import os
import re
import glob

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()  # loads OPENAI_API_KEY from .env

DATA_DIR = "data"
DB_DIR = "chroma_store"


# 1. LOAD ---- read each transcript, throw away the VTT timestamps
def load_transcripts():

    docs = []
    for path in glob.glob(f"{DATA_DIR}/*.vtt"):
        lines = []
        for line in open(path):
            line = line.strip()
            if not line or line == "WEBVTT" or "-->" in line:
                continue
            lines.append(line)
        text = " ".join(lines)

        session = re.search(r"Session[ _]*(\d+)", path).group(1)

        docs.append(Document(page_content=text, metadata={"session": session}))

    return docs


# 2. BUILD ---- chunk, embed once, and keep it on disk so we don't re-embed
def load_store():
    api_key = os.environ.get("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=api_key or "sk-placeholder-key-for-init",
    )

    if os.path.exists(DB_DIR):
        return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    if not api_key or api_key.startswith("sk-placeholder"):
        raise ValueError(
            "OPENAI_API_KEY is not set. Please add your OPENAI_API_KEY to .env before building the Chroma vector store."
        )

    docs = load_transcripts()

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    ).split_documents(docs)

    return Chroma.from_documents(chunks, embeddings, persist_directory=DB_DIR)



from collections import defaultdict
from rank_bm25 import BM25Okapi

_cached_chunks: list[Document] | None = None
_bm25_index: BM25Okapi | None = None


def get_all_chunks() -> list[Document]:
    """Retrieve all document chunks for BM25 indexing, cached in memory."""
    global _cached_chunks
    if _cached_chunks is not None:
        return _cached_chunks

    # Attempt to load existing chunks from Chroma without re-embedding
    try:
        store = load_store()
        data = store._collection.get(include=["documents", "metadatas"])
        if data and data.get("documents"):
            _cached_chunks = [
                Document(page_content=d, metadata=m or {})
                for d, m in zip(data["documents"], data["metadatas"])
            ]
            return _cached_chunks
    except Exception:
        pass

    # Fallback to parsing and splitting raw transcripts
    docs = load_transcripts()
    _cached_chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    ).split_documents(docs)
    return _cached_chunks


def get_bm25_index() -> BM25Okapi:
    """Lazily construct and cache the BM25 Okapi index."""
    global _bm25_index
    if _bm25_index is None:
        chunks = get_all_chunks()
        tokenized_corpus = [doc.page_content.lower().split() for doc in chunks]
        _bm25_index = BM25Okapi(tokenized_corpus)
    return _bm25_index


def bm25_retrieve(query: str, k: int = 10) -> list[Document]:
    """Retrieve top-k documents using BM25 sparse keyword matching."""
    chunks = get_all_chunks()
    if not chunks:
        return []

    index = get_bm25_index()
    tokenized_query = query.lower().split()
    scores = index.get_scores(tokenized_query)

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            doc = chunks[idx]
            doc_copy = Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "bm25_score": float(scores[idx])}
            )
            results.append(doc_copy)
    return results


def reciprocal_rank_fusion(
    dense_results: list[Document],
    sparse_results: list[Document],
    k: int = 60,
    top_n: int = 10,
) -> list[Document]:
    """
    Combine dense and sparse retrieval results using Reciprocal Rank Fusion.
    RRF score = sum(1 / (k + rank_i + 1))
    """
    doc_scores: dict[str, float] = defaultdict(float)
    doc_map: dict[str, Document] = {}

    for rank, doc in enumerate(dense_results):
        key = doc.page_content[:200].strip()
        doc_scores[key] += 1.0 / (k + rank + 1)
        doc_map[key] = doc

    for rank, doc in enumerate(sparse_results):
        key = doc.page_content[:200].strip()
        doc_scores[key] += 1.0 / (k + rank + 1)
        if key not in doc_map:
            doc_map[key] = doc

    sorted_keys = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
    return [doc_map[key] for key in sorted_keys[:top_n]]


def hybrid_retrieve(query: str | list[str], fetch_k: int = 10) -> list[Document]:
    """
    Perform hybrid retrieval (Chroma Dense Vector + BM25 Sparse Keyword)
    fused with Reciprocal Rank Fusion (RRF).
    """
    queries = [query] if isinstance(query, str) else query

    # 1. Dense retrieval
    dense_results = retrieve_queries(queries, k=fetch_k)

    # 2. Sparse BM25 retrieval
    sparse_results = []
    seen_sparse = set()
    for q in queries:
        for doc in bm25_retrieve(q, k=fetch_k):
            key = doc.page_content[:200].strip()
            if key not in seen_sparse:
                seen_sparse.add(key)
                sparse_results.append(doc)

    # 3. Fuse via RRF
    return reciprocal_rank_fusion(dense_results, sparse_results, top_n=fetch_k)


def build_retriever(k: int = 5):
    return load_store().as_retriever(search_kwargs={"k": k})


def retrieve_queries(queries: list[str], k: int = 10) -> list[Document]:
    """Retrieve documents across multiple expanded/rewritten queries, deduplicating by content."""
    store = load_store()
    seen_contents = set()
    all_docs = []

    for q in queries:
        docs = store.similarity_search(q, k=k)
        for doc in docs:
            # Use the first 200 characters as a deduplication key
            key = doc.page_content[:200].strip()
            if key not in seen_contents:
                seen_contents.add(key)
                all_docs.append(doc)

    return all_docs


def get_corpus_summary() -> str:
    """Return summary of the document corpus covered by course transcripts."""
    return (
        "Course transcripts covering LLM evaluations and RAG systems across 8 sessions:\n"
        "- Session 1-2: Foundations of LLM evaluation, offline vs online evaluation, LLM-as-a-judge.\n"
        "- Session 3-4: Evaluation metrics, DeepEval, RAG triad (Faithfulness, Answer Relevancy, Contextual Relevancy).\n"
        "- Session 5-6: Component-level testing (retriever recall/precision, generator faithfulness), golden datasets, synthesizer.\n"
        "- Session 7-8: Operational evaluations (latency, cost, reliability), safety gates (toxicity, leakage, scope), and regression testing suites with baseline comparison."
    )


# 3. TRY IT ---- python src/retriever.py
if __name__ == "__main__":

    retriever = build_retriever()

    results = retriever.invoke("what is regression testing?")
    
    for r in results:
        print(f"[Session {r.metadata['session']}] {r.page_content[:150]}...\n")
