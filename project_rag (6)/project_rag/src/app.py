"""
src/app.py — Streamlit chat UI for the RAG pipeline.

Run from the PROJECT ROOT (the chroma_store/ and data/ paths are relative):

    uv run streamlit run src/app.py
"""

import sys
import time
from pathlib import Path

# streamlit puts src/ on sys.path, not the project root — add the root so that
# `from src.rag_pipeline import ...` resolves the same way it does in the evals.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.rag_pipeline import RagPipeline

ABSTENTION = "I don't have enough information in the course material to answer that."

st.set_page_config(page_title="LLM Evals TA", page_icon="🎓", layout="centered")


# the pipeline loads the vector store + downloads/loads the cross-encoder, so
# build it once per (fetch_k, top_k) combo instead of on every rerun
@st.cache_resource(show_spinner="Loading retriever and reranker…")
def get_pipeline(fetch_k: int, top_k: int) -> RagPipeline:
    return RagPipeline(fetch_k=fetch_k, top_k=top_k)


with st.sidebar:
    st.header("Retrieval settings")
    fetch_k = st.slider(
        "fetch_k — candidates from the vector store",
        min_value=5,
        max_value=30,
        value=10,
        help="How many chunks the bi-encoder over-retrieves before reranking.",
    )
    top_k = st.slider(
        "top_k — chunks kept after reranking",
        min_value=1,
        max_value=10,
        value=5,
        help="How many chunks the cross-encoder keeps and feeds to the generator.",
    )
    if top_k > fetch_k:
        st.warning("top_k is larger than fetch_k — you'll get at most fetch_k chunks.")

    st.divider()
    self_corrective = st.toggle(
        "Self-Corrective Strategy Loop",
        value=True,
        help="Runs document relevance filtering, IsSUP hallucination reflection/revision, and IsUSE query expansion loop."
    )
    retrieval_mode_opt = st.selectbox(
        "Retrieval Routing",
        options=["Auto (Decision Node)", "Dense Only", "Hybrid (Dense + BM25)"],
        index=0,
        help="Auto uses the Decision Node to choose between dense and hybrid BM25 retrieval."
    )
    show_context = st.toggle("Show retrieved context", value=True)
    show_reflection = st.toggle("Show reflection details", value=True)
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(
        "Answers are grounded only in the course transcripts. If the material "
        "doesn't cover it, the assistant abstains."
    )

force_mode = None
if "Dense Only" in retrieval_mode_opt:
    force_mode = "dense"
elif "Hybrid" in retrieval_mode_opt:
    force_mode = "hybrid"


st.title("🎓 LLM Evals — Self-Corrective TA")
st.caption("Ask anything about the course transcripts. Answers are self-verified and grounded.")

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_reflection_details(msg: dict) -> None:
    """Show self-reflection diagnostics (IsSUP, IsUSE, retries)."""
    issup = msg.get("issup")
    isuse = msg.get("isuse")
    retries = msg.get("retries", 0)
    rewrite_tries = msg.get("rewrite_tries", 0)
    evidence = msg.get("evidence", [])
    decision = msg.get("retrieval_decision")

    if not (issup or isuse or decision):
        return

    with st.expander("🔍 Self-Correction Diagnostics"):
        if decision:
            st.markdown(f"**Decision Node:** Retrieve: `{decision.get('should_retrieve')}` | Mode: `{decision.get('mode')}`")
            if decision.get("reason"):
                st.caption(f"_{decision['reason']}_")
            st.divider()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Groundedness (IsSUP):** `{issup}`")
            st.markdown(f"**Revision Loops:** `{retries}`")
        with col2:
            st.markdown(f"**Usefulness (IsUSE):** `{isuse}`")
            st.markdown(f"**Query Rewrites:** `{rewrite_tries}`")

        if evidence:
            st.markdown("**Grounding Evidence Quotes:**")
            for e in evidence:
                st.markdown(f"- _{e}_")


def render_context(context: list[str], latency: float | None = None) -> None:
    """Show the retrieved chunks behind an answer in a collapsed expander."""
    label = f"📚 {len(context)} retrieved chunk{'s' if len(context) != 1 else ''}"
    if latency is not None:
        label += f" · {latency:.1f}s"
    with st.expander(label):
        for i, chunk in enumerate(context, start=1):
            st.markdown(f"**Chunk {i}**")
            st.text(chunk)
            if i != len(context):
                st.divider()


# replay the conversation so far
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            if show_reflection:
                render_reflection_details(message)
            if show_context and message.get("context"):
                render_context(message["context"], message.get("latency"))


query = st.chat_input("e.g. What is the difference between online and offline eval?")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        try:
            rag = get_pipeline(fetch_k, top_k)
            with st.spinner("Retrieving, evaluating & self-correcting…"):
                started = time.perf_counter()
                result = rag.invoke(
                    query,
                    self_corrective=self_corrective,
                    force_mode=force_mode,
                )
                latency = time.perf_counter() - started
        except Exception as exc:  # bad API key, missing store, model download, …
            st.error(f"Something went wrong: {exc}")
            st.session_state.messages.pop()  # drop the unanswered turn
        else:
            answer, context = result["answer"], result["context"]
            st.markdown(answer)
            if answer.strip() == ABSTENTION:
                st.info("The assistant abstained — this isn't covered in the transcripts.")
            if show_reflection:
                render_reflection_details(result)
            if show_context:
                render_context(context, latency)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "context": context,
                    "latency": latency,
                    "retrieval_decision": result.get("retrieval_decision"),
                    "issup": result.get("issup"),
                    "isuse": result.get("isuse"),
                    "evidence": result.get("evidence", []),
                    "retries": result.get("retries", 0),
                    "rewrite_tries": result.get("rewrite_tries", 0),
                }
            )


