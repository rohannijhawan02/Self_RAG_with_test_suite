from src.reranker import RerankingRetriever, decide_retrieval
from src.generator import (
    generate,
    generate_direct,
    generate_with_reflection,
    check_isuse,
    expand_queries,
    rewrite_with_feedback,
    ABSTENTION,
)
from langsmith import traceable


class RagPipeline:
    def __init__(
        self,
        fetch_k: int = 10,
        top_k: int = 5,
        max_issup_retries: int = 3,
        max_rewrite_tries: int = 3,
    ):
        # one retriever instance — loads the store + reranker model once
        self.retriever = RerankingRetriever(fetch_k=fetch_k, top_k=top_k)
        self.fetch_k = fetch_k
        self.top_k = top_k
        self.max_issup_retries = max_issup_retries
        self.max_rewrite_tries = max_rewrite_tries

    @traceable(run_type="chain", name="RagPipeline")
    def invoke(
        self,
        query: str,
        self_corrective: bool = True,
        force_mode: str | None = None,
    ) -> dict:
        """
        Execute the RAG pipeline.

        Decision Node & Strategy Flow:
          0. Decision Node: Evaluates if retrieval is needed (should_retrieve),
             and routes to 'dense' (conceptual) or 'hybrid' (BM25 + Dense RRF).
          1. Retrieval: Retrieves candidates using selected mode and reranks with CrossEncoder.
          2. Relevance Filter: LLM filters out off-topic chunks before generation.
          3. Grounded Generation + IsSUP Loop: Verifies claims against context quotes (inner loop).
          4. IsUSE Usefulness Check: Validates answering intent. On failure, triggers
             Query Expansion (retry 1) or Feedback Rewrite (retry 2+) up to max_rewrite_tries.
        """
        # --- 0. Decision Node ---
        try:
            decision = decide_retrieval(query)
            should_retrieve = decision.should_retrieve
            retrieval_mode = force_mode or decision.retrieval_mode
            decision_reason = decision.reason
        except Exception as exc:
            # Defensive fallback if decision LLM call errors
            should_retrieve = True
            retrieval_mode = force_mode or "hybrid"
            decision_reason = f"Fallback to retrieval due to: {exc}"

        decision_meta = {
            "should_retrieve": should_retrieve,
            "mode": retrieval_mode if should_retrieve else "none",
            "reason": decision_reason,
        }

        # If general greeting / no retrieval needed, answer directly
        if not should_retrieve:
            answer = generate_direct(query)
            return {
                "query": query,
                "context": [],
                "answer": answer,
                "retrieval_decision": decision_meta,
                "issup": "fully_supported",
                "isuse": "useful",
                "evidence": [],
                "retries": 0,
                "rewrite_tries": 0,
            }

        if not self_corrective:
            # Baseline single-pass execution
            docs = self.retriever.invoke(query, mode=retrieval_mode)
            context = [doc.page_content for doc in docs]
            answer = generate(query, context)
            return {
                "query": query,
                "context": context,
                "answer": answer,
                "retrieval_decision": decision_meta,
                "issup": "unverified",
                "isuse": "unverified",
                "retries": 0,
                "rewrite_tries": 0,
            }

        # --- Self-Corrective Loop ---
        current_queries = [query]
        previous_queries = []
        rewrite_tries = 0
        last_answer = ""
        last_reason = ""

        while rewrite_tries <= self.max_rewrite_tries:
            # 1. RETRIEVE (Dense or Hybrid BM25+Dense) + RERANK
            candidates = self.retriever.invoke(current_queries, mode=retrieval_mode)

            # 2. RELEVANCE FILTER (LLM checks topic relevance)
            relevant_docs = self.retriever.filter_relevant(query, candidates)

            if not relevant_docs:
                # No relevant documents found
                if rewrite_tries < self.max_rewrite_tries:
                    rewrite_tries += 1
                    if rewrite_tries == 1:
                        current_queries = expand_queries(
                            query,
                            last_answer,
                            "Initial retrieval did not contain relevant context."
                        )
                    else:
                        current_queries = rewrite_with_feedback(
                            query,
                            previous_queries,
                            last_answer,
                            "Retrieved documents were judged irrelevant to the query."
                        )
                    previous_queries.extend(current_queries)
                    continue
                else:
                    return {
                        "query": query,
                        "context": [],
                        "answer": ABSTENTION,
                        "retrieval_decision": decision_meta,
                        "issup": "fully_supported",
                        "isuse": "not_useful",
                        "evidence": [],
                        "retries": 0,
                        "rewrite_tries": rewrite_tries,
                    }

            context = [doc.page_content for doc in relevant_docs]

            # 3. GENERATION + INNER IsSUP GROUNDEDNESS REFLECTION LOOP
            reflection_res = generate_with_reflection(
                query, context, max_retries=self.max_issup_retries
            )
            answer = reflection_res["answer"]
            issup = reflection_res["issup"]
            issup_retries = reflection_res["retries"]
            evidence = reflection_res["evidence"]

            # If the answer is an explicit abstention, return immediately
            if answer.strip() == ABSTENTION:
                return {
                    "query": query,
                    "context": context,
                    "answer": answer,
                    "retrieval_decision": decision_meta,
                    "issup": issup,
                    "isuse": "useful",
                    "evidence": evidence,
                    "retries": issup_retries,
                    "rewrite_tries": rewrite_tries,
                }

            # 4. OUTER IsUSE USEFULNESS CHECK
            use_dec = check_isuse(query, answer)

            if use_dec.isuse == "useful" or rewrite_tries >= self.max_rewrite_tries:
                return {
                    "query": query,
                    "context": context,
                    "answer": answer,
                    "retrieval_decision": decision_meta,
                    "issup": issup,
                    "isuse": use_dec.isuse,
                    "use_reason": use_dec.reason,
                    "evidence": evidence,
                    "retries": issup_retries,
                    "rewrite_tries": rewrite_tries,
                }

            # If not useful and retries remain, reformulate query
            rewrite_tries += 1
            last_answer = answer
            last_reason = use_dec.reason

            if rewrite_tries == 1:
                current_queries = expand_queries(query, last_answer, last_reason)
            else:
                current_queries = rewrite_with_feedback(
                    query, previous_queries, last_answer, last_reason
                )
            previous_queries.extend(current_queries)

        # Fallback return
        return {
            "query": query,
            "context": context,
            "answer": answer if 'answer' in locals() else ABSTENTION,
            "retrieval_decision": decision_meta,
            "issup": issup if 'issup' in locals() else "no_support",
            "isuse": "not_useful",
            "retries": issup_retries if 'issup_retries' in locals() else 0,
            "rewrite_tries": rewrite_tries,
        }


# quick manual smoke test: python -m src.rag_pipeline
if __name__ == "__main__":
    rag = RagPipeline()
    result = rag.invoke("What is the difference between offline and online evaluation?")
    print("QUERY:             ", result["query"])
    print("RETRIEVAL DECISION:", result.get("retrieval_decision"))
    print("ANSWER:            ", result["answer"])
    print("IsSUP:             ", result.get("issup"))
    print("IsUSE:             ", result.get("isuse"))
    print("REWRITE TRIES:     ", result.get("rewrite_tries"))
    print("IsSUP RETRIES:     ", result.get("retries"))
    print("\nCONTEXT CHUNKS:")
    for i, chunk in enumerate(result.get("context", [])):
        print(f"  [{i}] {chunk[:120]}...")
