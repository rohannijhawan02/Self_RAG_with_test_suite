from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from src.retriever import load_store, retrieve_queries, hybrid_retrieve, get_corpus_summary
from langsmith import traceable

# small, fast, CPU-friendly reranker. Downloads once (~80MB) on first run.
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RetrieveDecision(BaseModel):
    should_retrieve: bool = Field(
        ...,
        description="True if course materials or technical concepts are needed to answer. False for general greetings or non-technical chat."
    )
    retrieval_mode: Literal["dense", "hybrid"] = Field(
        default="dense",
        description=(
            "'hybrid' (dense vector + BM25 keyword matching) for queries with specific metric names, acronyms, formulas, or codes. 'dense' for conceptual questions."
        )
    )
    reason: str = Field(..., description="Short explanation of the decision.")


decide_retrieval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You decide whether to retrieve course materials to answer a student's question.\n\n"
            "AVAILABLE COURSE MATERIAL TOPICS:\n{corpus_summary}\n\n"
            "Decision Rules:\n"
            "1. should_retrieve:\n"
            "   - True if the question asks about LLM evaluations, RAG, metrics, testing, or course topics.\n"
            "   - False if it's a general greeting (e.g. 'hi', 'hello'), introduction ('who are you?'), or non-course small talk.\n"
            "2. retrieval_mode:\n"
            "   - 'hybrid' (dense + BM25 keyword): use when the query has specific acronyms (TTFT, PII, BLEU, ROUGE), metric names, formulas, or technical codes.\n"
            "   - 'dense' (semantic vector): use for broader conceptual, comparative, or explanatory questions."
        ),
        ("human", "Question: {question}"),
    ]
)


class RelevanceDecision(BaseModel):
    is_relevant: bool = Field(
        ...,
        description="True if the document contains info relevant to the question topic or concept."
    )


relevance_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are judging document relevance at a TOPIC level.\n"
            "A document is relevant if it discusses the same concept, entity, or topic area as the question.\n"
            "It does NOT need to contain the complete or exact answer (grounding is verified later).\n"
            "When in doubt, return is_relevant=true."
        ),
        ("human", "Question:\n{question}\n\nDocument:\n{document}"),
    ]
)


def decide_retrieval(query: str, corpus_summary: str | None = None) -> RetrieveDecision:
    """Decide whether to retrieve course material and whether to use dense or hybrid mode."""
    api_key = os.environ.get("OPENAI_API_KEY")
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=api_key or "sk-placeholder-key-for-init",
    )
    summary = corpus_summary or get_corpus_summary()
    structured_llm = llm.with_structured_output(RetrieveDecision)
    return structured_llm.invoke(
        decide_retrieval_prompt.format_messages(
            question=query,
            corpus_summary=summary,
        )
    )



class RerankingRetriever:
    def __init__(self, fetch_k: int = 10, top_k: int = 5):
        self._store = None
        self._reranker = None
        self.fetch_k = fetch_k   # how many the bi-encoder brings back (over-retrieve)
        self.top_k = top_k       # how many survive after reranking
        self._llm = None

    @property
    def store(self):
        if self._store is None:
            self._store = load_store()
        return self._store

    @property
    def reranker(self):
        if self._reranker is None:
            self._reranker = CrossEncoder(CROSS_ENCODER)
        return self._reranker

    @property
    def llm(self):
        if self._llm is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            self._llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
                api_key=api_key or "sk-placeholder-key-for-init",
            )
        return self._llm



    def rerank(self, query: str, candidates: list[Document], top_k: int = None) -> list[Document]:
        """Rerank candidates using the cross-encoder."""
        if not candidates:
            return []
        if len(candidates) == 1:
            return candidates

        k = top_k or self.top_k
        pairs = [(query, doc.page_content) for doc in candidates]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:k]]

    def filter_relevant(self, query: str, docs: list[Document]) -> list[Document]:
        """Filter retrieved documents by topic-level relevance using LLM."""
        if not docs:
            return []

        structured_llm = self.llm.with_structured_output(RelevanceDecision)
        relevant_docs = []
        for doc in docs:
            try:
                decision: RelevanceDecision = structured_llm.invoke(
                    relevance_prompt.format_messages(
                        question=query,
                        document=doc.page_content[:1500],
                    )
                )
                if decision.is_relevant:
                    relevant_docs.append(doc)
            except Exception:
                # If LLM check fails, keep document defensively
                relevant_docs.append(doc)

        return relevant_docs

    @traceable(run_type="retriever", name="RerankingRetriever")
    def invoke(
        self,
        query,
        filter_relevance: bool = False,
        mode: Literal["dense", "hybrid"] = "dense",
    ):
        """Retrieve candidates (dense or hybrid BM25) and rerank down to top_k."""
        eval_query = query[0] if isinstance(query, list) and query else (query if isinstance(query, str) else "")

        if mode == "hybrid":
            candidates = hybrid_retrieve(query, fetch_k=self.fetch_k)
        else:
            if isinstance(query, list):
                candidates = retrieve_queries(query, k=self.fetch_k)
            else:
                candidates = self.store.similarity_search(query, k=self.fetch_k)

        reranked = self.rerank(eval_query, candidates, top_k=self.top_k)

        if filter_relevance:
            return self.filter_relevant(eval_query, reranked)

        return reranked
