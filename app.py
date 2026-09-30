"""Minimal Streamlit frontend for the RAG API."""

from __future__ import annotations

import os
from pathlib import Path

import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
UPLOAD_DIR = Path("data/uploads")


def _api_error(response: requests.Response) -> str:
    try:
        return response.json().get("detail", "Request failed.")
    except ValueError:
        return f"Request failed with status {response.status_code}."


st.set_page_config(page_title="RAG Knowledge QA", page_icon="📚", layout="wide")
st.title("RAG Knowledge Base QA")

with st.sidebar:
    st.header("Add a document")
    uploaded_file = st.file_uploader("Choose a document", type=["txt", "pdf", "srt", "vtt"])
    if st.button("Index document", disabled=uploaded_file is None, use_container_width=True):
        try:
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            destination = UPLOAD_DIR / Path(uploaded_file.name).name
            destination.write_bytes(uploaded_file.getvalue())
            response = requests.post(f"{API_URL}/index", json={"path": str(destination)}, timeout=120)
            if response.ok:
                details = response.json()
                st.success(f"Indexed {details['chunks_added']} chunks.")
            else:
                st.error(_api_error(response))
        except requests.RequestException as exc:
            st.error(f"Could not reach the API: {exc}")
        except OSError as exc:
            st.error(f"Could not save the upload: {exc}")

    st.divider()
    st.header("Indexed sources")
    try:
        source_response = requests.get(f"{API_URL}/sources", timeout=10)
        if source_response.ok:
            for source in source_response.json():
                st.write(f"- {source}")
        else:
            st.caption(_api_error(source_response))
    except requests.RequestException:
        st.caption("API is not available.")

st.header("Ask a question")
question = st.text_input("Question", placeholder="Ask something about your indexed documents")
k = st.slider("Chunks to retrieve", min_value=1, max_value=10, value=5)

if st.button("Ask", type="primary", disabled=not question.strip()):
    try:
        response = requests.post(f"{API_URL}/query", json={"query": question, "k": k}, timeout=120)
        if response.ok:
            result = response.json()
            if result.get("used_fallback"):
                st.warning(result["answer"])
            else:
                st.subheader("Answer")
                st.write(result["answer"])

            with st.expander("Sources", expanded=True):
                citations = result.get("citations", [])
                if not citations:
                    st.caption("No cited sources.")
                for citation in citations:
                    st.markdown(f"**{citation.get('marker', '')} {citation.get('source', '')}**")
                    if citation.get("start") is not None:
                        st.caption(f"Timestamp: {citation['start']}s - {citation['end']}s")
                    st.write(citation.get("text", ""))
        else:
            st.error(_api_error(response))
    except requests.RequestException as exc:
        st.error(f"Could not reach the API: {exc}")
