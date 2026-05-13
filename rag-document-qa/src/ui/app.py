"""
Streamlit UI for the RAG Document Q&A system.

Run with:
    streamlit run src/ui/app.py
"""

import os
import httpx
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Enterprise Document Q&A",
    page_icon="📄",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Configuration")

default_api_url = os.getenv("API_URL", "http://localhost:8000")
api_url = st.sidebar.text_input(
    "API URL",
    value=default_api_url,
    help="Base URL of the RAG API server.",
)

collection = st.sidebar.text_input(
    "Collection Name",
    value="my-docs",
    help="ChromaDB collection to query or ingest into.",
)

top_k = st.sidebar.slider(
    "Top-K Chunks",
    min_value=1,
    max_value=10,
    value=5,
    help="Number of context chunks to retrieve per query.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("📤 Upload Document")

uploaded_file = st.sidebar.file_uploader(
    "Upload PDF, TXT, or DOCX",
    type=["pdf", "txt", "docx"],
    help="File will be sent to the API for ingestion.",
)

if uploaded_file is not None:
    if st.sidebar.button("Ingest Document", type="primary"):
        with st.sidebar:
            with st.spinner("Ingesting document…"):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    data = {"collection": collection}
                    
                    response = httpx.post(
                        f"{api_url}/ingest",
                        files=files,
                        data=data,
                        timeout=120.0,
                    )
                    response.raise_for_status()
                    data = response.json()
                    st.success(
                        f"✅ Ingested **{data['chunks_stored']}** chunks from "
                        f"`{uploaded_file.name}` into collection `{data['collection']}`."
                    )
                except httpx.HTTPStatusError as exc:
                    st.error(f"API error {exc.response.status_code}: {exc.response.text}")
                except Exception as exc:
                    st.error(f"Ingestion failed: {exc}")

# ---------------------------------------------------------------------------
# Main area — Q&A
# ---------------------------------------------------------------------------
st.title("📄 Enterprise Document Q&A")
st.caption(
    "Ask questions about your ingested documents. "
    "Answers are grounded in retrieved context and checked for hallucinations."
)

question = st.text_area(
    "Your Question",
    placeholder="e.g. What is the maximum allowed remote work days per week according to the HR policy?",
    height=100,
)

submit = st.button("🔍 Ask", type="primary", use_container_width=True)

if submit:
    if not question.strip():
        st.warning("Please enter a question before submitting.")
    else:
        with st.spinner("Retrieving context and generating answer…"):
            try:
                response = httpx.post(
                    f"{api_url}/query",
                    json={
                        "question": question,
                        "collection": collection,
                        "top_k": top_k,
                    },
                    timeout=60.0,
                )
                response.raise_for_status()
                data = response.json()

                # ----------------------------------------------------------
                # Answer
                # ----------------------------------------------------------
                st.markdown("### 💬 Answer")

                if data.get("blocked"):
                    st.error(
                        "⛔ This response was **blocked** by the hallucination guardrail.\n\n"
                        + data["answer"]
                    )
                else:
                    st.markdown(data["answer"])

                # ----------------------------------------------------------
                # Faithfulness badge
                # ----------------------------------------------------------
                score = data.get("faithfulness_score", 0.0)
                latency = data.get("latency_ms", 0)

                col1, col2, col3 = st.columns(3)
                with col1:
                    if score >= 0.7:
                        st.success(f"✅ Faithfulness: **{score:.2f}**")
                    else:
                        st.error(f"⚠️ Faithfulness: **{score:.2f}**")
                with col2:
                    st.info(f"⏱️ Latency: **{latency} ms**")
                with col3:
                    blocked_label = "🔴 Blocked" if data.get("blocked") else "🟢 Passed"
                    st.info(f"Guardrail: **{blocked_label}**")

                # ----------------------------------------------------------
                # Sources table
                # ----------------------------------------------------------
                sources = data.get("sources", [])
                if sources:
                    st.markdown("### 📚 Sources")
                    import pandas as pd

                    df = pd.DataFrame(sources)
                    df.columns = [c.capitalize() for c in df.columns]
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("No source documents were retrieved.")

            except httpx.HTTPStatusError as exc:
                st.error(
                    f"API returned error {exc.response.status_code}: {exc.response.text}"
                )
            except httpx.ConnectError:
                st.error(
                    f"Could not connect to the API at `{api_url}`. "
                    "Make sure the server is running."
                )
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "RAG Document Q&A · Powered by LangChain, ChromaDB, Gemini & RAGAS"
)
