import os
import json
import tempfile
from pathlib import Path
from datetime import datetime

import streamlit as st

from medical_ocr_pipeline import MedicalDocumentPipeline, VectorDBManager


st.set_page_config(page_title="Medical Document OCR", page_icon="🏥", layout="wide")

st.sidebar.image("https://img.icons8.com/color/96/000000/medical-doctor.png", width=80)
st.sidebar.title("⚙️ Configuration")

gemini_key = st.sidebar.text_input(
    "🔑 Gemini API Key (optional)",
    type="password",
    help="Leave blank to skip LLM enrichment",
)
vector_db_path = st.sidebar.text_input("💾 Vector DB Path", value="./medical_vector_db")
output_dir = st.sidebar.text_input("📁 Output Directory", value="./output")

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "processed_docs" not in st.session_state:
    st.session_state.processed_docs = []

if st.sidebar.button("🚀 Initialize Pipeline"):
    try:
        key = gemini_key.strip() or None
        st.session_state.pipeline = MedicalDocumentPipeline(
            gemini_api_key=key, vector_db_path=vector_db_path
        )
        st.success("✅ Pipeline initialized successfully")
    except Exception as e:
        st.session_state.pipeline = None
        st.error(f"❌ Initialization failed: {e}")

st.title("🏥 Medical Document OCR Pipeline")
st.markdown(
    """
Upload PDFs or images — files will be processed and a **PROVER JSON** will be saved to the output folder.  
The full JSON is **not displayed** on-screen for privacy.
"""
)

if not st.session_state.pipeline:
    st.warning("Please initialize the pipeline from the sidebar to proceed.")
    st.stop()

tabs = st.tabs(["📤 Upload & Process", "📋 Processed Summary", "🔎 Search"])

# 📤 Upload & Process Tab

with tabs[0]:
    uploaded = st.file_uploader(
        "Upload PDF / Image files (multiple)",
        type=["pdf", "png", "jpg", "jpeg", "tiff"],
        accept_multiple_files=True,
    )

    if uploaded:
        if st.button("🔄 Process files"):
            prog = st.progress(0)
            status = st.empty()
            for i, uf in enumerate(uploaded, start=1):
                status.text(f"Processing ({i}/{len(uploaded)}): {uf.name}")
                suffix = Path(uf.name).suffix
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp.write(uf.getbuffer())
                tmp.flush()
                tmp.close()
                try:
                    res = st.session_state.pipeline.process_document(
                        tmp.name, output_dir=output_dir
                    )

                    # store minimal summary in session
                    summary = {
                        "original_filename": uf.name,
                        "saved_prover_path": res.get("prover_json_path"),
                        "document_header": res.get("document_header"),
                        "provenance": res.get("provenance"),
                        "quality_metrics": res.get("quality_metrics"),
                        "processed_at": datetime.utcnow().isoformat(),
                    }
                    st.session_state.processed_docs.insert(0, summary)
                    st.success(f"✅ Processed: {uf.name}")
                except Exception as e:
                    st.error(f"❌ Error processing {uf.name}: {e}")
                finally:
                    if os.path.exists(tmp.name):
                        os.unlink(tmp.name)
                prog.progress(i / len(uploaded))

            status.text("✅ All files processed.")
            st.balloons()

# 📋 Processed Summary Tab
with tabs[1]:
    st.header("Processed Documents (Summary)")
    if not st.session_state.processed_docs:
        st.info("No documents processed yet.")
    else:
        for idx, doc in enumerate(st.session_state.processed_docs):
            with st.expander(f"{doc['original_filename']} — processed at {doc['processed_at']}"):
                hdr = doc.get("document_header") or {}
                qm = doc.get("quality_metrics") or {}

                if not hdr:
                    st.warning(
                        "⚠️ No document header found — the OCR or LLM may have failed."
                    )
                else:
                    st.write("**Document ID:**", hdr.get("document_id", "N/A"))
                    st.write("**Source file:**", hdr.get("source_file", "N/A"))
                    st.write("**Pages:**", hdr.get("page_count", "N/A"))

                overall_conf = qm.get("overall_confidence", None)
                if overall_conf is not None:
                    st.metric("Overall Confidence", f"{overall_conf * 100:.1f}%")
                else:
                    st.info("No confidence score available.")

                st.write("**Saved PROVER JSON:**", doc.get("saved_prover_path") or "N/A")

                path = doc.get("saved_prover_path")
                if path and os.path.exists(path):
                    with open(path, "rb") as fh:
                        st.download_button(
                            label=f"📥 Download {os.path.basename(path)}",
                            data=fh,
                            file_name=os.path.basename(path),
                            mime="application/json",
                            key=f"dl_{idx}",
                        )
                else:
                    st.error("Saved PROVER JSON file not found.")
# 🔎 Search Tab
with tabs[2]:
    st.header("Semantic Search (Vector DB)")
    q = st.text_input("Enter search query")
    k = st.slider("Results", min_value=1, max_value=10, value=5)
    if st.button("🔎 Search"):
        try:
            results = st.session_state.pipeline.vector_db.search(q, k)
            if not results:
                st.info("No results found — vector DB may be empty.")
            else:
                for i, r in enumerate(results, start=1):
                    st.write(f"**Result {i} — score: {r.get('score', 0):.4f}**")
                    doc_txt = r.get("document", "")
                    st.write(doc_txt[:800] + ("..." if len(doc_txt) > 800 else ""))
        except Exception as e:
            st.error(f"Search failed: {e}")

st.markdown("---")
st.caption(
    "⚠️ Full PROVER JSONs are saved to disk (output folder). Download them for detailed review."
)
