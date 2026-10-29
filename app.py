"""Streamlit UI for the RAG Knowledge Base QA API."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from html import escape
from pathlib import Path

import requests
import streamlit as st
import streamlit.components.v1 as components


API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
UPLOAD_DIR = Path("data/uploads")
API_HEADERS = {
    "X-API-Key": os.getenv("API_ACCESS_TOKEN", "").strip(),
}


def _api_error(response: requests.Response) -> str:
    try:
        return response.json().get("detail", "Request failed.")
    except ValueError:
        return f"Request failed with status {response.status_code}."


def _get_sources() -> list[str]:
    try:
        response = requests.get(f"{API_URL}/sources", headers=API_HEADERS, timeout=10)
        return response.json() if response.ok else []
    except requests.RequestException:
        return []


def _backend_connected() -> bool:
    try:
        response = requests.get(f"{API_URL}/health", headers=API_HEADERS, timeout=3)
        return response.ok and response.json().get("status") == "ok"
    except (requests.RequestException, ValueError):
        return False


def _file_icon(filename: str) -> str:
    return {".pdf": "📄", ".txt": "📄", ".srt": "🎞️", ".vtt": "🎞️"}.get(Path(filename).suffix.lower(), "📄")


def _file_meta(filename: str) -> tuple[str, str]:
    try:
        stat = (UPLOAD_DIR / Path(filename).name).stat()
        return f"{stat.st_size / 1024:.1f} KB", datetime.fromtimestamp(stat.st_mtime).strftime("%b %d, %Y")
    except OSError:
        return "—", "Not indexed"


def _link_citations(answer: str) -> str:
    safe_answer = escape(answer).replace("\n", "<br>")
    return re.sub(r"\[(\d+)\]", r'<a href="#source-\1">[\1]</a>', safe_answer)


def _copy_answer(answer: str) -> None:
    text = json.dumps(answer)
    components.html(
        f"""<button onclick='navigator.clipboard.writeText({text}).then(() => this.innerText = "Copied ✓")'
        style='border:1px solid #d4def5;border-radius:8px;background:#fff;color:#21458f;padding:6px 10px;cursor:pointer;font-size:12px;'>Copy answer</button>""",
        height=38,
    )


for key, default in {
    "document_status": {},
    "document_chunks": {},
    "removed_uploads": set(),
    "last_result": None,
    "query_rewrite": os.getenv("USE_QUERY_REWRITE", "false").lower() in {"1", "true", "yes", "on"},
    "reranking": os.getenv("USE_RERANKER", "false").lower() in {"1", "true", "yes", "on"},
}.items():
    st.session_state.setdefault(key, default)


st.set_page_config(page_title="RAG Knowledge Base QA", page_icon="▣", layout="wide")
st.markdown(
    """
    <style>
    :root { --navy:#14254b; --blue:#2364f5; --muted:#60739c; --line:#dce5f6; }
    html, body, [class*="css"] { font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
    .stApp { background:linear-gradient(135deg,#f8fbff 0%,#eef5ff 100%); color:var(--navy); }
    .block-container { max-width:1440px; padding:1rem 1.15rem 1.4rem; }
    section[data-testid="stSidebar"] { width:360px !important; background:rgba(255,255,255,.92); border-right:1px solid #dce5f6; }
    section[data-testid="stSidebar"] > div:first-child { width:360px !important; padding:1rem .95rem; }
    div[data-testid="stVerticalBlockBorderWrapper"] { background:rgba(255,255,255,.92); border:1px solid var(--line); border-radius:10px; box-shadow:0 2px 8px rgba(32,64,125,.045); padding:5px 9px; }
    .brand { display:flex; gap:.55rem; align-items:center; margin:.1rem .2rem 1rem; }
    .brand-mark { width:31px; height:31px; display:grid; place-items:center; border-radius:8px; background:linear-gradient(135deg,#2f7bff,#1550d3); color:#fff; font-size:1.15rem; }
    .brand-name { color:var(--navy); font-size:1.08rem; font-weight:800; line-height:1.05; }
    .brand-subtitle { color:#6e80a6; font-size:.72rem; margin-top:.18rem; }
    .side-nav { display:flex; align-items:center; gap:.6rem; padding:.52rem .65rem; border-radius:8px; color:#526486; font-size:.86rem; font-weight:650; margin:.12rem 0; }
    .side-nav.active { background:#e9f0ff; color:#1554d0; }
    section[data-testid="stSidebar"] .card-title { font-size:.87rem; }
    section[data-testid="stSidebar"] label { font-size:.76rem !important; }
    section[data-testid="stSidebar"] .stCaption { font-size:.68rem; }
    .card-title { color:var(--navy); font-size:.94rem; font-weight:780; margin-bottom:.15rem; }
    .card-subtitle { color:var(--muted); font-size:.73rem; margin-bottom:.45rem; }
    .page-title { color:var(--navy); font-size:1.95rem; font-weight:800; margin:0; }
    .page-subtitle { color:#62769f; font-size:.88rem; margin:.15rem 0 .6rem; }
    .connection { float:right; border:1px solid #cfe9df; background:#f1fcf8; color:#276550; border-radius:999px; padding:.35rem .7rem; font-size:.75rem; }
    [data-testid="stFileUploaderDropzone"] { border:1px dashed #b9cbed; border-radius:9px; background:#fbfdff; }
    .doc-row { display:flex; gap:.5rem; align-items:flex-start; padding:.42rem 0; border-bottom:1px solid #edf1f8; }
    .doc-icon { width:26px; height:26px; display:grid; place-items:center; border-radius:7px; background:#eff4ff; flex:0 0 auto; font-size:.9rem; }
    .doc-main { min-width:0; flex:1; }
    .doc-name { color:#233657; font-size:.78rem; line-height:1.3; overflow-wrap:anywhere; }
    .doc-meta { color:#7183a8; font-size:.68rem; margin-top:.2rem; }
    .doc-date { color:#7183a8; font-size:.67rem; white-space:nowrap; }
    .pill { display:inline-block; border-radius:999px; padding:2px 8px; font-size:10px; font-weight:750; }
    .pill-indexed { color:#087a4e; background:#dff8eb; }.pill-processing { color:#866300; background:#fff2bd; }.pill-failed { color:#a43030; background:#ffe4e4; }
    .stButton > button { border-radius:8px; border:1px solid #cbd9f2; color:#1c3971; min-height:2.15rem; padding:.25rem .55rem; font-size:.78rem; }
    .stButton > button[kind="primary"] { background:#2364f5; color:#fff; border-color:#2364f5; }
    .upload-card button[kind="secondary"] { background:#2364f5 !important; color:#fff !important; border-color:#2364f5 !important; }
    .stTextArea textarea, .stTextInput input { border-radius:9px; border-color:#cbd9f2; background:#fff; font-size:.86rem; }
    span[data-baseweb="tag"] { background:#3d73eb !important; color:#fff !important; }
    .answer-text { color:#17233f; font-size:.9rem; line-height:1.48; }.answer-text a { color:#1d5de6; font-weight:800; text-decoration:none; }
    .reference { display:flex; gap:.5rem; padding:.42rem 0; border-top:1px solid #edf1f8; }.reference-name { color:#263b66; font-size:.78rem; font-weight:750; overflow-wrap:anywhere; }.reference-meta,.metric-label { color:#7183a8; font-size:.68rem; }.reference-snippet { color:#526486; font-size:.75rem; line-height:1.4; margin-top:.25rem; }
    .metric { background:#fbfdff; border:1px solid #e0e8f7; border-radius:9px; padding:.5rem .6rem; }.metric-value { color:#1b3265; font-size:1.1rem; font-weight:750; margin-top:.2rem; }
    .empty { background:#fbfdff; border:1px dashed #d2def2; border-radius:9px; color:#7385a7; padding:.8rem; text-align:center; font-size:.76rem; }
    .upload-gap { height: 18px; }
    </style>
    """,
    unsafe_allow_html=True,
)


indexed_sources = _get_sources()
for source in indexed_sources:
    st.session_state.document_status.setdefault(source, "Indexed")

with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-mark">▱</div><div><div class="brand-name">RAG Knowledge QA</div><div class="brand-subtitle">Ask questions from your documents</div></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="side-nav active">▤ &nbsp; Documents</div><div class="side-nav">⚙ &nbsp; Settings</div>', unsafe_allow_html=True)

    # Upload card intentionally appears before the indexed document list.
    uploaded_files = []
    with st.container(border=True):
        st.markdown('<div class="card-title">⇧ &nbsp; Upload Document</div><div class="card-subtitle">Add PDF, TXT, SRT, or VTT files</div>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader("Choose PDF, TXT, SRT, or VTT documents", type=["pdf", "txt", "srt", "vtt"], accept_multiple_files=True, label_visibility="collapsed", key="top_upload")
        visible_uploads = [file for file in uploaded_files if file.name not in st.session_state.removed_uploads]
        if visible_uploads:
            for file in visible_uploads:
                st.caption(f"{_file_icon(file.name)}  {file.name}")
        if st.button("⇧  Upload Document", type="primary", use_container_width=True, disabled=not visible_uploads, key="top_upload_button"):
            st.info("Files selected. Click Index Document to add them to the knowledge base.")
        if st.button("⌘  Index Document", use_container_width=True, disabled=not visible_uploads, key="top_index_button"):
            for file in visible_uploads:
                filename = Path(file.name).name
                st.session_state.document_status[filename] = "Processing"
                try:
                    with st.spinner(f"Indexing {filename}..."):
                        response = requests.post(
                            f"{API_URL}/index/upload",
                            files={"file": (filename, file.getvalue(), file.type or "application/octet-stream")},
                            headers=API_HEADERS,
                            timeout=120,
                        )
                    details = response.json() if response.ok else {}
                    if response.ok and details.get("chunks_added", 0) > 0:
                        st.session_state.document_status[filename] = "Indexed"
                        st.session_state.document_chunks[filename] = details["chunks_added"]
                    else:
                        st.session_state.document_status[filename] = "Failed"
                        st.error(_api_error(response) if not response.ok else f"No text extracted from {filename}.")
                except (requests.RequestException, OSError) as exc:
                    st.session_state.document_status[filename] = "Failed"
                    st.error(f"Could not index {filename}: {exc}")
            st.rerun()

    with st.container(border=True):
        st.markdown(f'<div class="card-title">▣ &nbsp; Documents <span style="float:right;background:#e8eef9;border-radius:999px;padding:2px 8px;font-size:.7rem;">{len(indexed_sources)}</span></div>', unsafe_allow_html=True)
        sidebar_sources = st.multiselect("Search in", indexed_sources, default=indexed_sources, key="sidebar_sources")
        if indexed_sources:
            for source in indexed_sources:
                size, date = _file_meta(source)
                status = st.session_state.document_status.get(source, "Indexed")
                pill = {"Indexed":"pill-indexed", "Processing":"pill-processing", "Failed":"pill-failed"}.get(status, "pill-indexed")
                chunks = st.session_state.document_chunks.get(source, "—")
                st.markdown(f'<div class="doc-row"><div class="doc-icon">{_file_icon(source)}</div><div class="doc-main"><div class="doc-name" title="{escape(source, quote=True)}">{escape(source)}</div><div class="doc-meta">{size} · {chunks} chunks · <span class="pill {pill}">{status}</span></div></div><div class="doc-date">⋮</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="empty">No documents indexed yet</div>', unsafe_allow_html=True)

    # The upload card is rendered before the indexed documents card.
    if False:
        st.markdown('<div class="card-title">⇧ &nbsp; Upload Document</div><div class="card-subtitle">Add PDF, TXT, SRT, or VTT files</div>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader("Choose PDF, TXT, SRT, or VTT documents", type=["pdf", "txt", "srt", "vtt"], accept_multiple_files=True, label_visibility="collapsed")
        visible_uploads = [file for file in uploaded_files if file.name not in st.session_state.removed_uploads]
        if visible_uploads:
            for file in visible_uploads:
                st.caption(f"{_file_icon(file.name)}  {file.name}")
        if st.button("⇧  Upload Document", type="primary", use_container_width=True, disabled=not visible_uploads):
            st.info("Files selected. Click Index Document to add them to the knowledge base.")
        if st.button("⌘  Index Document", use_container_width=True, disabled=not visible_uploads):
            for file in visible_uploads:
                filename = Path(file.name).name
                st.session_state.document_status[filename] = "Processing"
                try:
                    with st.spinner(f"Indexing {filename}..."):
                        response = requests.post(
                            f"{API_URL}/index/upload",
                            files={"file": (filename, file.getvalue(), file.type or "application/octet-stream")},
                            headers=API_HEADERS,
                            timeout=120,
                        )
                    details = response.json() if response.ok else {}
                    if response.ok and details.get("chunks_added", 0) > 0:
                        st.session_state.document_status[filename] = "Indexed"
                        st.session_state.document_chunks[filename] = details["chunks_added"]
                    else:
                        st.session_state.document_status[filename] = "Failed"
                        st.error(_api_error(response) if not response.ok else f"No text extracted from {filename}.")
                except (requests.RequestException, OSError) as exc:
                    st.session_state.document_status[filename] = "Failed"
                    st.error(f"Could not index {filename}: {exc}")
            st.rerun()

    with st.container(border=True):
        st.markdown('<div class="card-title">⚙ &nbsp; Settings</div>', unsafe_allow_html=True)
        st.session_state.query_rewrite = st.toggle("Use query rewriting", value=st.session_state.query_rewrite)
        st.caption("Rewrites your question for better results")
        st.session_state.reranking = st.toggle("Use reranking", value=st.session_state.reranking)
        st.caption("Improves relevance using a cross-encoder")


header_cols = st.columns([5, 1.5])
with header_cols[0]:
    st.markdown('<div class="page-title">✦ RAG Knowledge Base QA</div><div class="page-subtitle">Upload documents. Ask questions. Get answers with sources.</div>', unsafe_allow_html=True)
with header_cols[1]:
    status = '<span style="color:#16a765;">●</span> Backend Connected' if _backend_connected() else '<span style="color:#d64545;">●</span> Backend Offline'
    st.markdown(f'<div class="connection">{status}</div>', unsafe_allow_html=True)

with st.container(border=True):
    st.markdown('<div class="card-title">▣ &nbsp; Select Document</div>', unsafe_allow_html=True)
    select_cols = st.columns([5, 1.4])
    with select_cols[0]:
        selected_main = st.selectbox("Document", ["All documents"] + indexed_sources, label_visibility="collapsed")
    with select_cols[1]:
        st.button("▣  View Indexed Documents", use_container_width=True)

with st.container(border=True):
    st.markdown('<div class="card-title">☷ &nbsp; Your Question</div>', unsafe_allow_html=True)
    question_cols = st.columns([5.8, 1])
    with question_cols[0]:
        question = st.text_area("Question", placeholder="Ask a question about your indexed documents...", height=68, label_visibility="collapsed")
    with question_cols[1]:
        ask = st.button("➤  Ask", type="primary", use_container_width=True, disabled=not question.strip())
    st.markdown('<div class="card-subtitle" style="margin:.35rem 0 .1rem;">Number of chunks to retrieve</div>', unsafe_allow_html=True)
    k = st.slider("Chunks", min_value=1, max_value=10, value=5, label_visibility="collapsed")

if ask:
    selected_for_query = [selected_main] if selected_main != "All documents" else sidebar_sources
    payload = {"query": question.strip(), "k": int(k), "rewrite": bool(st.session_state.query_rewrite)}
    if selected_for_query:
        payload["sources"] = selected_for_query
    try:
        with st.spinner("Retrieving sources and generating your answer..."):
            response = requests.post(f"{API_URL}/query", json=payload, headers=API_HEADERS, timeout=120)
        if response.ok:
            st.session_state.last_result = response.json()
            st.rerun()
        else:
            st.error(_api_error(response))
    except requests.Timeout:
        st.error("The request timed out. Try fewer chunks or a shorter question.")
    except requests.RequestException as exc:
        st.error(f"Could not reach the API: {exc}")

result = st.session_state.last_result
with st.container(border=True):
    st.markdown('<div class="card-title">✦ &nbsp; Answer</div>', unsafe_allow_html=True)
    if result:
        _copy_answer(result.get("answer", ""))
        st.markdown(f'<div class="answer-text">{_link_citations(result.get("answer", ""))}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="empty">▧<br><br><b>Your answer will appear here</b><br><small>Ask a question to see the response with sources.</small></div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title" style="margin-top:.8rem;">▣ &nbsp; Sources</div>', unsafe_allow_html=True)
    citations = result.get("citations", []) if result else []
    if citations:
        for index, citation in enumerate(citations, start=1):
            source = str(citation.get("source", "Unknown source"))
            location = f"Page {citation['page']}" if citation.get("page") is not None else f"Chunk {citation.get('chunk_id', '—')}"
            snippet = escape(str(citation.get("text", ""))[:320])
            st.markdown(f'<div class="reference" id="source-{index}"><div class="doc-icon">{_file_icon(source)}</div><div><div class="reference-name">[{index}] {escape(source)}</div><div class="reference-meta">{location}</div><div class="reference-snippet">{snippet}</div></div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="empty">Sources will be shown here after you ask a question.</div>', unsafe_allow_html=True)

with st.container(border=True):
    st.markdown('<div class="card-title">◷ &nbsp; Performance Metrics</div>', unsafe_allow_html=True)
    metrics = result.get("timing_ms", {}) if result else {}
    metric_cols = st.columns(3)
    for col, label, key in zip(metric_cols, ("Retrieval Time", "Generation Time", "Total Time"), ("retrieval", "generation", "total")):
        with col:
            value = f"{metrics[key] / 1000:.2f} s" if metrics.get(key) is not None else "—"
            note = "Measured from latest query" if metrics.get(key) is not None else "Not available yet"
            st.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-label">{note}</div></div>', unsafe_allow_html=True)
