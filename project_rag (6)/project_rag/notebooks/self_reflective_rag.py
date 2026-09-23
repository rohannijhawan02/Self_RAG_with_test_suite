# %% [markdown]
# # Self-Reflective RAG Pipeline
#
# A comprehensive, self-reflective Retrieval-Augmented Generation pipeline
# built with **LangChain + LangGraph**.
#
# ### Pipeline Flow:
# 1. Generate document structure summary from titles/headings
# 2. Decide: retrieval needed? If yes → dense or hybrid? If no → direct answer or web search?
# 3. Take top 7 results and check relevance via LLM
# 4. If ≥1 relevant → generate answer from context
# 5. IsSUP: check if answer is grounded in context (self-loop to revise)
# 6. IsUSE: check if answer actually addresses the question
# 7. If not useful → retry 1: query expansion (5 queries) / retry 2+: rewrite with feedback
# 8. If max retries exceeded or no docs → return "No answer found"

# %% Cell 1: Imports & Setup
# =============================================================================
# All imports, API key placeholders, and LLM/embeddings initialization
# =============================================================================

import os
import re
import math
from typing import List, TypedDict, Literal, Optional, Dict, Any
from collections import defaultdict

from pydantic import BaseModel, Field

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.tools.tavily_search import TavilySearchResults

from langgraph.graph import StateGraph, START, END

from sentence_transformers import CrossEncoder

# ── API Key Placeholders ─────────────────────────────────────────────────────
# Replace these with your actual API keys before running
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
os.environ["TAVILY_API_KEY"] = "YOUR_TAVILY_API_KEY"

# ── LLM & Embeddings ─────────────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


# %% Cell 2: Document Loading & Hierarchical Recursive Chunking
# =============================================================================
# Load PDFs and chunk using recursive hierarchical splitting
# =============================================================================

DOCUMENT_DIR = "./documents/"
MAX_CHUNK_SIZE = 600
CHUNK_OVERLAP = 150

# ── Heading-aware hierarchical separators ─────────────────────────────────────
# Order matters: split by largest headings first, then smaller, then paragraphs
HIERARCHICAL_SEPARATORS = [
    "\n# ",       # H1
    "\n## ",      # H2
    "\n### ",     # H3
    "\n#### ",    # H4
    "\n\n",       # Paragraph break
    "\n",         # Line break
    ". ",         # Sentence boundary
    " ",          # Word boundary
]


def extract_headings_from_text(text: str) -> List[str]:
    """Extract heading-like lines from document text."""
    headings = []
    lines = text.split("\n")
    for line in lines:
        stripped = line.strip()
        # Match markdown headings
        if re.match(r"^#{1,4}\s+", stripped):
            headings.append(stripped)
        # Match lines that look like titles (short, capitalized, no ending punctuation)
        elif (
            len(stripped) > 3
            and len(stripped) < 100
            and stripped[0].isupper()
            and not stripped.endswith(".")
            and not stripped.endswith(",")
            and stripped.upper() == stripped  # ALL CAPS = likely heading
        ):
            headings.append(stripped)
    return headings


def hierarchical_recursive_chunk(
    documents: List[Document],
    max_chunk_size: int = MAX_CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    Recursively chunk documents with hierarchical awareness.

    Strategy:
    1. First try splitting by the largest heading level
    2. If a chunk is still too large, recursively split by smaller headings
    3. Fall back to RecursiveCharacterTextSplitter for leaf chunks
    """
    final_chunks: List[Document] = []

    base_splitter = RecursiveCharacterTextSplitter(
        separators=HIERARCHICAL_SEPARATORS,
        chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    for doc in documents:
        text = doc.page_content
        metadata = doc.metadata.copy()

        # Extract headings for metadata enrichment
        headings = extract_headings_from_text(text)
        if headings:
            metadata["headings"] = headings[:5]  # Store up to 5 headings

        # If already small enough, keep as-is
        if len(text) <= max_chunk_size:
            final_chunks.append(
                Document(page_content=text, metadata=metadata)
            )
            continue

        # Use the recursive splitter with hierarchical separators
        sub_chunks = base_splitter.split_text(text)

        for i, chunk_text in enumerate(sub_chunks):
            chunk_meta = metadata.copy()
            chunk_meta["chunk_index"] = i
            chunk_meta["total_chunks"] = len(sub_chunks)

            # Try to identify the heading context for this chunk
            # Look for the nearest heading above this chunk in the original text
            chunk_start = text.find(chunk_text[:50])  # approximate position
            if chunk_start >= 0:
                preceding_text = text[:chunk_start]
                preceding_headings = extract_headings_from_text(preceding_text)
                if preceding_headings:
                    chunk_meta["section_heading"] = preceding_headings[-1]

            final_chunks.append(
                Document(page_content=chunk_text, metadata=chunk_meta)
            )

    return final_chunks


# ── Load documents ────────────────────────────────────────────────────────────
def load_documents(doc_dir: str = DOCUMENT_DIR) -> List[Document]:
    """Load all PDFs from the document directory."""
    import glob

    pdf_files = glob.glob(os.path.join(doc_dir, "*.pdf"))
    all_docs = []
    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(pdf_path)
            all_docs.extend(loader.load())
        except Exception as e:
            print(f"Warning: Failed to load {pdf_path}: {e}")
    return all_docs


# Load and chunk
raw_docs = load_documents()
chunks = hierarchical_recursive_chunk(raw_docs)
print(f"Loaded {len(raw_docs)} raw document pages → {len(chunks)} chunks")


# %% Cell 3: Document Structure Summary Generation
# =============================================================================
# Generate an overall summary of the available document structure
# =============================================================================

def generate_document_summary(documents: List[Document], llm_instance) -> str:
    """
    Generate a concise summary of the document corpus structure
    by extracting and summarizing titles/headings.
    """
    # Collect all headings and source info
    all_headings: List[str] = []
    sources: set = set()

    for doc in documents:
        source = doc.metadata.get("source", "Unknown")
        sources.add(os.path.basename(source))

        headings = extract_headings_from_text(doc.page_content)
        all_headings.extend(headings)

    # Deduplicate headings while preserving order
    seen = set()
    unique_headings = []
    for h in all_headings:
        if h not in seen:
            seen.add(h)
            unique_headings.append(h)

    headings_text = "\n".join(unique_headings[:50])  # Cap at 50 headings
    sources_text = ", ".join(sorted(sources))

    summary_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a document analyst. Given a list of document sources and their "
                "section headings/titles, generate a concise summary (3-5 sentences) of "
                "what topics and information the document corpus covers.\n\n"
                "Focus on:\n"
                "- The main subject areas covered\n"
                "- The types of documents (policies, profiles, pricing, etc.)\n"
                "- Key topics a user could ask about\n\n"
                "Be specific and factual. Do not speculate about content not reflected in the headings."
            ),
            (
                "human",
                "Document Sources:\n{sources}\n\n"
                "Section Headings/Titles found:\n{headings}"
            ),
        ]
    )

    result = llm_instance.invoke(
        summary_prompt.format_messages(sources=sources_text, headings=headings_text)
    )
    return result.content


# Generate the document structure summary
document_summary = generate_document_summary(raw_docs, llm)
print("=== Document Structure Summary ===")
print(document_summary)


# %% Cell 4: Vector Store & Retriever Setup (FAISS + BM25)
# =============================================================================
# Set up both dense (FAISS) and sparse (BM25) retrievers
# =============================================================================

from rank_bm25 import BM25Okapi

# ── Dense Vector Store ────────────────────────────────────────────────────────
vector_store = FAISS.from_documents(chunks, embeddings)
dense_retriever = vector_store.as_retriever(search_kwargs={"k": 7})

# ── BM25 Sparse Index ────────────────────────────────────────────────────────
# Tokenize all chunks for BM25
chunk_texts = [doc.page_content for doc in chunks]
tokenized_chunks = [text.lower().split() for text in chunk_texts]
bm25_index = BM25Okapi(tokenized_chunks)


def bm25_retrieve(query: str, k: int = 7) -> List[Document]:
    """Retrieve top-k documents using BM25 sparse retrieval."""
    tokenized_query = query.lower().split()
    scores = bm25_index.get_scores(tokenized_query)

    # Get top-k indices
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:  # Only include docs with positive score
            doc = chunks[idx]
            # Store BM25 score in metadata for fusion
            doc_copy = Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "bm25_score": float(scores[idx])},
            )
            results.append(doc_copy)
    return results


def reciprocal_rank_fusion(
    dense_results: List[Document],
    sparse_results: List[Document],
    k: int = 60,
    top_n: int = 7,
) -> List[Document]:
    """
    Combine dense and sparse retrieval results using Reciprocal Rank Fusion.

    RRF score = sum(1 / (k + rank_i)) for each ranking list
    """
    # Map document content to scores and documents
    doc_scores: Dict[str, float] = defaultdict(float)
    doc_map: Dict[str, Document] = {}

    # Score from dense results
    for rank, doc in enumerate(dense_results):
        content_key = doc.page_content[:200]  # Use first 200 chars as key
        doc_scores[content_key] += 1.0 / (k + rank + 1)
        doc_map[content_key] = doc

    # Score from sparse results
    for rank, doc in enumerate(sparse_results):
        content_key = doc.page_content[:200]
        doc_scores[content_key] += 1.0 / (k + rank + 1)
        if content_key not in doc_map:
            doc_map[content_key] = doc

    # Sort by RRF score and return top_n
    sorted_keys = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
    return [doc_map[key] for key in sorted_keys[:top_n]]


print(f"Vector store: {vector_store.index.ntotal} vectors indexed")
print(f"BM25 index: {len(tokenized_chunks)} documents indexed")


# %% Cell 5: Graph State Definition
# =============================================================================
# Define the state that flows through the LangGraph
# =============================================================================

class State(TypedDict):
    question: str
    document_summary: str           # Overall doc structure summary

    # Retrieval routing
    retrieval_mode: Literal["dense", "hybrid", "none"]
    retrieval_query: str
    expanded_queries: List[str]     # For query expansion
    rewrite_tries: int

    need_retrieval: bool
    use_web_search: bool            # Whether to use Tavily web search
    web_query: str

    # Retrieved documents
    docs: List[Document]
    relevant_docs: List[Document]
    context: str
    answer: str

    # Post-generation verification (IsSUP)
    issup: Literal["fully_supported", "partially_supported", "no_support"]
    evidence: List[str]
    retries: int                    # IsSUP revision retries

    # Usefulness check (IsUSE)
    isuse: Literal["useful", "not_useful"]
    use_reason: str


# %% Cell 6: Retrieval Decision Node (3-way routing)
# =============================================================================
# Decide whether to retrieve, and if so how (dense/hybrid), or go direct/web
# =============================================================================

class RetrieveDecision(BaseModel):
    should_retrieve: bool = Field(
        ...,
        description="True if internal documents are needed to answer reliably."
    )
    retrieval_mode: Literal["dense", "hybrid"] = Field(
        default="dense",
        description=(
            "If should_retrieve is True: 'dense' for semantic search only, "
            "'hybrid' for both semantic and keyword-based search. "
            "Use 'hybrid' when the question contains specific terms, names, or codes "
            "that benefit from exact keyword matching."
        ),
    )
    use_web_search: bool = Field(
        default=False,
        description=(
            "True if the question is about recent events, external facts, or topics "
            "clearly outside the scope of internal documents. "
            "Only True when should_retrieve is False."
        ),
    )


decide_retrieval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You decide how to answer a question based on the available document corpus.\n\n"
            "AVAILABLE DOCUMENTS SUMMARY:\n{document_summary}\n\n"
            "You must decide:\n"
            "1. should_retrieve: True if internal documents likely contain relevant info.\n"
            "2. retrieval_mode: 'dense' (semantic similarity) or 'hybrid' (semantic + keyword BM25).\n"
            "   - Use 'hybrid' when the question has specific names, codes, product names, "
            "policy numbers, or technical terms.\n"
            "   - Use 'dense' for general conceptual questions.\n"
            "3. use_web_search: True ONLY if should_retrieve is False AND the question is "
            "about external/recent facts not covered by internal docs.\n\n"
            "Guidelines:\n"
            "- If the question is about topics in the document summary → should_retrieve=True\n"
            "- If it's a general knowledge question → should_retrieve=False, use_web_search=False\n"
            "- If it's about current events or external info → should_retrieve=False, use_web_search=True\n"
            "- When unsure, choose should_retrieve=True, retrieval_mode='hybrid'"
        ),
        ("human", "Question: {question}"),
    ]
)

should_retrieve_llm = llm.with_structured_output(RetrieveDecision)


def decide_retrieval(state: State):
    """Node: Decide retrieval strategy based on question and document summary."""
    decision: RetrieveDecision = should_retrieve_llm.invoke(
        decide_retrieval_prompt.format_messages(
            question=state["question"],
            document_summary=state.get("document_summary", "No summary available."),
        )
    )
    return {
        "need_retrieval": decision.should_retrieve,
        "retrieval_mode": decision.retrieval_mode if decision.should_retrieve else "none",
        "use_web_search": decision.use_web_search and not decision.should_retrieve,
    }


def route_after_decide(state: State) -> Literal["generate_direct", "retrieve_dense", "retrieve_hybrid", "rewrite_web_query"]:
    """Route based on retrieval decision."""
    if not state["need_retrieval"]:
        if state.get("use_web_search"):
            return "rewrite_web_query"
        return "generate_direct"

    mode = state.get("retrieval_mode", "dense")
    if mode == "hybrid":
        return "retrieve_hybrid"
    return "retrieve_dense"


# %% Cell 7: Direct Answer Node
# =============================================================================
# Generate answer from general knowledge (no retrieval)
# =============================================================================

direct_generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer using only your general knowledge.\n"
            "If it requires specific company/internal info, say:\n"
            "'I don't know based on my general knowledge.'"
        ),
        ("human", "{question}"),
    ]
)


def generate_direct(state: State):
    """Node: Generate direct answer without retrieval."""
    out = llm.invoke(
        direct_generation_prompt.format_messages(question=state["question"])
    )
    return {"answer": out.content}


# %% Cell 8: Tavily Web Search Node
# =============================================================================
# Rewrite query for web search and search via Tavily
# =============================================================================

class WebQuery(BaseModel):
    query: str = Field(
        ...,
        description="Web search query composed of keywords, 6-14 words."
    )


rewrite_web_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user question into a web search query composed of keywords.\n"
            "Rules:\n"
            "- Keep it short (6–14 words).\n"
            "- If the question implies recency, add (last 30 days).\n"
            "- Do NOT answer the question.\n"
            "- Return JSON with a single key: query",
        ),
        ("human", "Question: {question}"),
    ]
)

rewrite_web_chain = rewrite_web_prompt | llm.with_structured_output(WebQuery)

tavily = TavilySearchResults(max_results=5)


def rewrite_web_query(state: State):
    """Node: Rewrite question into a web search query."""
    out = rewrite_web_chain.invoke({"question": state["question"]})
    return {"web_query": out.query}


def web_search(state: State):
    """Node: Execute Tavily web search and convert results to Documents."""
    q = state.get("web_query") or state["question"]
    results = tavily.invoke({"query": q})

    docs = []
    for r in results or []:
        title = r.get("title", "")
        url = r.get("url", "")
        content = r.get("content", "") or r.get("snippet", "")
        text = f"TITLE: {title}\nURL: {url}\nCONTENT:\n{content}"
        docs.append(
            Document(
                page_content=text,
                metadata={"source": "web", "url": url, "title": title},
            )
        )

    return {"docs": docs}


# %% Cell 9: Dense Retrieval Node
# =============================================================================
# Retrieve using FAISS dense vector search
# =============================================================================

def retrieve_dense(state: State):
    """Node: Dense retrieval via FAISS."""
    q = state.get("retrieval_query") or state["question"]

    # If we have expanded queries, retrieve for each and merge
    expanded = state.get("expanded_queries", [])
    if expanded:
        all_docs = []
        seen_contents = set()
        for eq in expanded:
            results = dense_retriever.invoke(eq)
            for doc in results:
                content_key = doc.page_content[:200]
                if content_key not in seen_contents:
                    seen_contents.add(content_key)
                    all_docs.append(doc)
        # Take top 7 by order (first queries are most important)
        return {"docs": all_docs[:7]}

    return {"docs": dense_retriever.invoke(q)}


# %% Cell 10: Hybrid Retrieval Node (Dense + BM25)
# =============================================================================
# Retrieve using both FAISS dense and BM25 sparse, fused with RRF
# =============================================================================

def retrieve_hybrid(state: State):
    """Node: Hybrid retrieval using Dense (FAISS) + Sparse (BM25) with RRF."""
    q = state.get("retrieval_query") or state["question"]

    # If we have expanded queries, run hybrid for each and merge
    expanded = state.get("expanded_queries", [])
    if expanded:
        all_docs = []
        seen_contents = set()
        for eq in expanded:
            dense_results = dense_retriever.invoke(eq)
            sparse_results = bm25_retrieve(eq, k=7)
            fused = reciprocal_rank_fusion(dense_results, sparse_results, top_n=7)
            for doc in fused:
                content_key = doc.page_content[:200]
                if content_key not in seen_contents:
                    seen_contents.add(content_key)
                    all_docs.append(doc)
        return {"docs": all_docs[:7]}

    # Single query hybrid retrieval
    dense_results = dense_retriever.invoke(q)
    sparse_results = bm25_retrieve(q, k=7)
    fused_results = reciprocal_rank_fusion(dense_results, sparse_results, top_n=7)
    return {"docs": fused_results}


# %% Cell 10b: Reranking Node
# =============================================================================
# Rerank retrieved documents using a cross-encoder model for better precision
# =============================================================================

# ── Cross-Encoder Reranker ────────────────────────────────────────────────────
# Uses a lightweight cross-encoder that scores (query, document) pairs directly.
# Much more accurate than bi-encoder similarity but slower — perfect for reranking
# a small candidate set (top-7) before the expensive LLM relevance check.
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
reranker_model = CrossEncoder(RERANK_MODEL_NAME)

RERANK_TOP_N = 7  # How many to keep after reranking


def rerank_documents(state: State):
    """
    Node: Rerank retrieved documents using a cross-encoder.

    Takes the candidate docs from retrieval (dense, hybrid, or web search)
    and re-scores them using a cross-encoder model that sees both the query
    and document together — producing much more accurate relevance scores
    than the initial bi-encoder retrieval.
    """
    docs = state.get("docs", [])
    question = state["question"]

    if not docs:
        return {"docs": []}

    # If only 1 doc, no need to rerank
    if len(docs) <= 1:
        return {"docs": docs}

    # Score each (question, document) pair with the cross-encoder
    pairs = [(question, doc.page_content) for doc in docs]
    scores = reranker_model.predict(pairs)

    # Sort by cross-encoder score descending
    scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

    # Keep top N reranked documents
    reranked = [doc for doc, score in scored_docs[:RERANK_TOP_N]]

    return {"docs": reranked}


# %% Cell 11: Relevance Filter Node
# =============================================================================
# Check each retrieved document for relevance using LLM
# =============================================================================

class RelevanceDecision(BaseModel):
    is_relevant: bool = Field(
        ...,
        description="True ONLY if the document contains info that can directly answer the question."
    )


is_relevant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are judging document relevance at a TOPIC level.\n"
            "Return JSON matching the schema.\n\n"
            "A document is relevant if it discusses the same entity or topic area as the question.\n"
            "It does NOT need to contain the exact answer.\n\n"
            "Examples:\n"
            "- HR policies are relevant to questions about notice period, probation, termination, benefits.\n"
            "- Pricing documents are relevant to questions about refunds, trials, billing terms.\n"
            "- Company profile is relevant to questions about leadership, culture, size, or strategy.\n\n"
            "Do NOT decide whether the document fully answers the question.\n"
            "That will be checked later by IsSUP.\n"
            "When unsure, return is_relevant=true."
        ),
        ("human", "Question:\n{question}\n\nDocument:\n{document}"),
    ]
)

relevance_llm = llm.with_structured_output(RelevanceDecision)


def check_relevance(state: State):
    """Node: Filter documents by LLM relevance check."""
    relevant_docs: List[Document] = []
    for doc in state.get("docs", []):
        decision: RelevanceDecision = relevance_llm.invoke(
            is_relevant_prompt.format_messages(
                question=state["question"],
                document=doc.page_content,
            )
        )
        if decision.is_relevant:
            relevant_docs.append(doc)
    return {"relevant_docs": relevant_docs}


def route_after_relevance(state: State) -> Literal["generate_from_context", "no_answer_found"]:
    """Route based on whether any relevant docs were found."""
    if state.get("relevant_docs") and len(state["relevant_docs"]) > 0:
        return "generate_from_context"
    return "no_answer_found"


# %% Cell 12: Generate from Context Node
# =============================================================================
# Generate answer from relevant context documents
# =============================================================================

rag_generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a business RAG chatbot.\n\n"
            "You will receive a CONTEXT block from internal company documents.\n"
            "Task:\n"
            "Answer the question based on the context.\n"
            "Don't mention that you are getting a context in your answer."
        ),
        ("human", "Question:\n{question}\n\nContext:\n{context}"),
    ]
)


def generate_from_context(state: State):
    """Node: Generate answer from relevant documents."""
    context = "\n\n---\n\n".join(
        [d.page_content for d in state.get("relevant_docs", [])]
    ).strip()

    if not context:
        return {"answer": "No answer found.", "context": ""}

    out = llm.invoke(
        rag_generation_prompt.format_messages(
            question=state["question"], context=context
        )
    )
    return {"answer": out.content, "context": context}


# %% Cell 13: IsSUP Groundedness Check Node
# =============================================================================
# Verify whether the answer is supported by the context (self-loop)
# =============================================================================

class IsSUPDecision(BaseModel):
    issup: Literal["fully_supported", "partially_supported", "no_support"]
    evidence: List[str] = Field(default_factory=list)


issup_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are verifying whether the ANSWER is supported by the CONTEXT.\n"
            "Return JSON with keys: issup, evidence.\n"
            "issup must be one of: fully_supported, partially_supported, no_support.\n\n"
            "How to decide issup:\n"
            "- fully_supported:\n"
            "  Every meaningful claim is explicitly supported by CONTEXT, and the ANSWER does NOT introduce\n"
            "  any qualitative/interpretive words that are not present in CONTEXT.\n"
            "  (Examples of disallowed words unless present in CONTEXT: culture, generous, robust, designed to,\n"
            "  supports professional development, best-in-class, employee-first, etc.)\n\n"
            "- partially_supported:\n"
            "  The core facts are supported, BUT the ANSWER includes ANY abstraction, interpretation, or qualitative\n"
            "  phrasing not explicitly stated in CONTEXT (e.g., calling policies 'culture', saying leave is 'generous',\n"
            "  or inferring outcomes like 'supports professional development').\n\n"
            "- no_support:\n"
            "  The key claims are not supported by CONTEXT.\n\n"
            "Rules:\n"
            "- Be strict: if you see ANY unsupported qualitative/interpretive phrasing, choose partially_supported.\n"
            "- If the answer is mostly unrelated to the question or unsupported, choose no_support.\n"
            "- Evidence: include up to 3 short direct quotes from CONTEXT that support the supported parts.\n"
            "- Do not use outside knowledge."
        ),
        (
            "human",
            "Question:\n{question}\n\n"
            "Answer:\n{answer}\n\n"
            "Context:\n{context}\n"
        ),
    ]
)

issup_llm = llm.with_structured_output(IsSUPDecision)

MAX_ISSUP_RETRIES = 3


def check_issup(state: State):
    """Node: Check if answer is grounded in context."""
    decision: IsSUPDecision = issup_llm.invoke(
        issup_prompt.format_messages(
            question=state["question"],
            answer=state.get("answer", ""),
            context=state.get("context", ""),
        )
    )
    return {"issup": decision.issup, "evidence": decision.evidence}


def route_after_issup(state: State) -> Literal["accept_answer", "revise_answer"]:
    """Route based on groundedness check."""
    # Fully supported → move to IsUSE
    if state.get("issup") == "fully_supported":
        return "accept_answer"

    # Max retries exceeded → accept whatever we have
    if state.get("retries", 0) >= MAX_ISSUP_RETRIES:
        return "accept_answer"

    return "revise_answer"


def accept_answer(state: State):
    """Node: Accept the current answer as-is."""
    return {}  # keep answer as-is


# %% Cell 14: Revise Answer Node
# =============================================================================
# Revise answer to be strictly grounded in context
# =============================================================================

revise_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a STRICT reviser.\n\n"
            "You must output based on the following format:\n\n"
            "FORMAT (quote-only answer):\n"
            "- <direct quote from the CONTEXT>\n"
            "- <direct quote from the CONTEXT>\n\n"
            "Rules:\n"
            "- Use ONLY the CONTEXT.\n"
            "- Do NOT add any new words besides bullet dashes and the quotes themselves.\n"
            "- Do NOT explain anything.\n"
            "- Do NOT say 'context', 'not mentioned', 'does not mention', 'not provided', etc.\n"
        ),
        (
            "human",
            "Question:\n{question}\n\n"
            "Current Answer:\n{answer}\n\n"
            "CONTEXT:\n{context}"
        ),
    ]
)


def revise_answer(state: State):
    """Node: Revise answer to be strictly grounded in context."""
    out = llm.invoke(
        revise_prompt.format_messages(
            question=state["question"],
            answer=state.get("answer", ""),
            context=state.get("context", ""),
        )
    )
    return {
        "answer": out.content,
        "retries": state.get("retries", 0) + 1,
    }


# %% Cell 15: IsUSE Usefulness Check Node
# =============================================================================
# Check if the answer actually addresses the user's question
# =============================================================================

class IsUSEDecision(BaseModel):
    isuse: Literal["useful", "not_useful"]
    reason: str = Field(..., description="Short reason in 1 line.")


isuse_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are judging USEFULNESS of the ANSWER for the QUESTION.\n\n"
            "Goal:\n"
            "- Decide if the answer actually addresses what the user asked.\n\n"
            "Return JSON with keys: isuse, reason.\n"
            "isuse must be one of: useful, not_useful.\n\n"
            "Rules:\n"
            "- useful: The answer directly answers the question or provides the requested specific info.\n"
            "- not_useful: The answer is generic, off-topic, or only gives related background without answering.\n"
            "- Do NOT use outside knowledge.\n"
            "- Do NOT re-check grounding (IsSUP already did that). Only check: 'Did we answer the question?'\n"
            "- Keep reason to 1 short line."
        ),
        (
            "human",
            "Question:\n{question}\n\nAnswer:\n{answer}"
        ),
    ]
)

isuse_llm = llm.with_structured_output(IsUSEDecision)

MAX_REWRITE_TRIES = 3  # Max outer loop retries


def check_isuse(state: State):
    """Node: Check if the answer is useful/addresses the question."""
    decision: IsUSEDecision = isuse_llm.invoke(
        isuse_prompt.format_messages(
            question=state["question"],
            answer=state.get("answer", ""),
        )
    )
    return {"isuse": decision.isuse, "use_reason": decision.reason}


def route_after_isuse(state: State) -> Literal["END", "expand_queries", "rewrite_with_feedback", "no_answer_found"]:
    """
    Route after usefulness check:
    - useful → END
    - not useful, retry 1 → query expansion (5 queries)
    - not useful, retry 2+ → rewrite with feedback
    - max retries exceeded → no_answer_found
    """
    if state.get("isuse") == "useful":
        return "END"

    current_tries = state.get("rewrite_tries", 0)

    if current_tries >= MAX_REWRITE_TRIES:
        return "no_answer_found"

    # First retry: use query expansion
    if current_tries == 0:
        return "expand_queries"

    # Subsequent retries: rewrite with feedback
    return "rewrite_with_feedback"


# %% Cell 16: Query Expansion Node
# =============================================================================
# Generate 5 diverse retrieval queries from the original question
# =============================================================================

class QueryExpansionResult(BaseModel):
    queries: List[str] = Field(
        ...,
        description="List of 5 diverse retrieval queries, each targeting a different aspect.",
        min_length=5,
        max_length=5,
    )


expand_queries_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a query expansion expert for retrieval-augmented generation.\n\n"
            "Given a user question, generate exactly 5 diverse retrieval queries.\n"
            "Each query should target a different aspect, angle, or reformulation of the question.\n\n"
            "Rules:\n"
            "- Keep each query short (6-16 words).\n"
            "- Include key entities from the original question.\n"
            "- Vary the phrasing: use synonyms, related terms, and different angles.\n"
            "- One query should be close to the original question.\n"
            "- One query should focus on specific keywords/entities.\n"
            "- One query should be a broader conceptual version.\n"
            "- Two queries should explore related sub-topics.\n"
            "- Do NOT answer the question.\n\n"
            "Return JSON with key: queries (list of 5 strings)"
        ),
        (
            "human",
            "Original Question:\n{question}\n\n"
            "Previous Answer (if any, which was not useful):\n{answer}\n\n"
            "Reason it was not useful:\n{reason}"
        ),
    ]
)

expand_llm = llm.with_structured_output(QueryExpansionResult)


def expand_queries(state: State):
    """Node: Expand original question into 5 diverse retrieval queries."""
    decision: QueryExpansionResult = expand_llm.invoke(
        expand_queries_prompt.format_messages(
            question=state["question"],
            answer=state.get("answer", ""),
            reason=state.get("use_reason", ""),
        )
    )

    return {
        "expanded_queries": decision.queries,
        "retrieval_query": decision.queries[0],  # Primary query
        "rewrite_tries": state.get("rewrite_tries", 0) + 1,
        # Reset retrieval state for clean pass
        "docs": [],
        "relevant_docs": [],
        "context": "",
        "retries": 0,  # Reset IsSUP retries
    }


# %% Cell 17: Query Rewrite with Feedback Node
# =============================================================================
# Rewrite queries using feedback from previous failed attempt
# =============================================================================

class RewriteWithFeedbackResult(BaseModel):
    queries: List[str] = Field(
        ...,
        description="List of 5 improved retrieval queries incorporating feedback.",
        min_length=5,
        max_length=5,
    )


rewrite_feedback_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are rewriting retrieval queries based on FEEDBACK from a failed attempt.\n\n"
            "The previous retrieval and answer generation did NOT produce a useful answer.\n"
            "You must generate 5 NEW, IMPROVED queries that address the failure.\n\n"
            "Rules:\n"
            "- Analyze WHY the previous answer was not useful (see the reason).\n"
            "- Generate queries that specifically target the MISSING information.\n"
            "- Use different keywords, synonyms, and angles than the previous queries.\n"
            "- Keep each query short (6-16 words).\n"
            "- Preserve key entities from the original question.\n"
            "- Do NOT answer the question.\n\n"
            "Return JSON with key: queries (list of 5 strings)"
        ),
        (
            "human",
            "Original Question:\n{question}\n\n"
            "Previous Retrieval Queries:\n{previous_queries}\n\n"
            "Previous Answer (not useful):\n{answer}\n\n"
            "Why it was not useful:\n{reason}"
        ),
    ]
)

rewrite_feedback_llm = llm.with_structured_output(RewriteWithFeedbackResult)


def rewrite_with_feedback(state: State):
    """Node: Rewrite queries using feedback from previous failed attempt."""
    previous_queries = state.get("expanded_queries", [])
    if not previous_queries:
        previous_queries = [state.get("retrieval_query", state["question"])]

    decision: RewriteWithFeedbackResult = rewrite_feedback_llm.invoke(
        rewrite_feedback_prompt.format_messages(
            question=state["question"],
            previous_queries="\n".join(previous_queries),
            answer=state.get("answer", ""),
            reason=state.get("use_reason", ""),
        )
    )

    return {
        "expanded_queries": decision.queries,
        "retrieval_query": decision.queries[0],
        "rewrite_tries": state.get("rewrite_tries", 0) + 1,
        # Reset retrieval state
        "docs": [],
        "relevant_docs": [],
        "context": "",
        "retries": 0,  # Reset IsSUP retries
    }


# %% Cell 18: No Answer Found Node
# =============================================================================
# Terminal node when no answer can be found
# =============================================================================

def no_answer_found(state: State):
    """Node: Return 'No answer found' when all retries are exhausted."""
    return {"answer": "No answer found.", "context": ""}


# %% Cell 19: Build LangGraph
# =============================================================================
# Wire all nodes together into a LangGraph StateGraph
# =============================================================================

graph = StateGraph(State)

# ── Add all nodes ─────────────────────────────────────────────────────────────
graph.add_node("decide_retrieval", decide_retrieval)
graph.add_node("generate_direct", generate_direct)
graph.add_node("rewrite_web_query", rewrite_web_query)
graph.add_node("web_search", web_search)
graph.add_node("retrieve_dense", retrieve_dense)
graph.add_node("retrieve_hybrid", retrieve_hybrid)
graph.add_node("rerank_documents", rerank_documents)
graph.add_node("check_relevance", check_relevance)
graph.add_node("generate_from_context", generate_from_context)
graph.add_node("check_issup", check_issup)
graph.add_node("accept_answer", accept_answer)
graph.add_node("revise_answer", revise_answer)
graph.add_node("check_isuse", check_isuse)
graph.add_node("expand_queries", expand_queries)
graph.add_node("rewrite_with_feedback", rewrite_with_feedback)
graph.add_node("no_answer_found", no_answer_found)

# ── Entry point ───────────────────────────────────────────────────────────────
graph.add_edge(START, "decide_retrieval")

# ── Retrieval decision routing (4-way) ────────────────────────────────────────
graph.add_conditional_edges(
    "decide_retrieval",
    route_after_decide,
    {
        "generate_direct": "generate_direct",
        "rewrite_web_query": "rewrite_web_query",
        "retrieve_dense": "retrieve_dense",
        "retrieve_hybrid": "retrieve_hybrid",
    },
)

# ── Direct answer → END ──────────────────────────────────────────────────────
graph.add_edge("generate_direct", END)

# ── Web search flow → rerank → relevance ─────────────────────────────────────
graph.add_edge("rewrite_web_query", "web_search")
graph.add_edge("web_search", "rerank_documents")

# ── Dense/Hybrid retrieval → rerank → relevance check ────────────────────────
graph.add_edge("retrieve_dense", "rerank_documents")
graph.add_edge("retrieve_hybrid", "rerank_documents")

# ── Rerank → relevance check ─────────────────────────────────────────────────
graph.add_edge("rerank_documents", "check_relevance")

# ── Relevance routing ────────────────────────────────────────────────────────
graph.add_conditional_edges(
    "check_relevance",
    route_after_relevance,
    {
        "generate_from_context": "generate_from_context",
        "no_answer_found": "no_answer_found",
    },
)

# ── Generate → IsSUP check ───────────────────────────────────────────────────
graph.add_edge("generate_from_context", "check_issup")

# ── IsSUP routing (self-loop for revision) ───────────────────────────────────
graph.add_conditional_edges(
    "check_issup",
    route_after_issup,
    {
        "accept_answer": "accept_answer",
        "revise_answer": "revise_answer",
    },
)

# ── Revise → re-check IsSUP (self-loop) ──────────────────────────────────────
graph.add_edge("revise_answer", "check_issup")

# ── Accept → IsUSE check ─────────────────────────────────────────────────────
graph.add_edge("accept_answer", "check_isuse")

# ── IsUSE routing ────────────────────────────────────────────────────────────
graph.add_conditional_edges(
    "check_isuse",
    route_after_isuse,
    {
        "END": END,
        "expand_queries": "expand_queries",
        "rewrite_with_feedback": "rewrite_with_feedback",
        "no_answer_found": "no_answer_found",
    },
)

# ── Query expansion/rewrite → back to retrieval ──────────────────────────────
# After expanding/rewriting, we need to re-enter retrieval
# Use the retrieval mode from the original decision
def route_after_expansion(state: State) -> Literal["retrieve_dense", "retrieve_hybrid"]:
    """Route back to the appropriate retrieval method after query expansion."""
    mode = state.get("retrieval_mode", "dense")
    if mode == "hybrid":
        return "retrieve_hybrid"
    return "retrieve_dense"


graph.add_conditional_edges(
    "expand_queries",
    route_after_expansion,
    {
        "retrieve_dense": "retrieve_dense",
        "retrieve_hybrid": "retrieve_hybrid",
    },
)

graph.add_conditional_edges(
    "rewrite_with_feedback",
    route_after_expansion,
    {
        "retrieve_dense": "retrieve_dense",
        "retrieve_hybrid": "retrieve_hybrid",
    },
)

# ── No answer → END ──────────────────────────────────────────────────────────
graph.add_edge("no_answer_found", END)

# ── Compile ───────────────────────────────────────────────────────────────────
app = graph.compile()
print("✅ Graph compiled successfully!")


# %% Cell 20: Test Execution
# =============================================================================
# Run the pipeline with a test query
# =============================================================================

def run_query(question: str) -> Dict[str, Any]:
    """Run a question through the self-reflective RAG pipeline."""
    initial_state = {
        "question": question,
        "document_summary": document_summary,
        "retrieval_mode": "dense",
        "retrieval_query": "",
        "expanded_queries": [],
        "rewrite_tries": 0,
        "need_retrieval": False,
        "use_web_search": False,
        "web_query": "",
        "docs": [],
        "relevant_docs": [],
        "context": "",
        "answer": "",
        "issup": "no_support",
        "evidence": [],
        "retries": 0,
        "isuse": "not_useful",
        "use_reason": "",
    }

    result = app.invoke(initial_state)
    return result


def print_result(result: Dict[str, Any]):
    """Pretty-print the pipeline result."""
    print("=" * 60)
    print(f"❓ Question: {result['question']}")
    print(f"📋 Retrieval Mode: {result.get('retrieval_mode', 'N/A')}")
    print(f"🔄 Rewrite Tries: {result.get('rewrite_tries', 0)}")
    print(f"🔁 IsSUP Retries: {result.get('retries', 0)}")
    print(f"✅ IsSUP: {result.get('issup', 'N/A')}")
    print(f"📊 IsUSE: {result.get('isuse', 'N/A')}")
    if result.get("use_reason"):
        print(f"   Reason: {result['use_reason']}")
    print(f"\n💬 Answer:\n{result.get('answer', 'No answer')}")
    if result.get("evidence"):
        print(f"\n📝 Evidence:")
        for e in result["evidence"]:
            print(f"   - {e}")
    print("=" * 60)


# ── Example: Run a test query ────────────────────────────────────────────────
# Uncomment and modify these to test:
#
# result = run_query("What is the notice period policy?")
# print_result(result)
#
# result = run_query("What is the capital of France?")
# print_result(result)
#
# result = run_query("What are the latest AI trends in 2024?")
# print_result(result)

print("\n🚀 Self-Reflective RAG Pipeline is ready!")
print("   Use run_query('your question') to test.")
print("   Use print_result(result) to see detailed output.")
