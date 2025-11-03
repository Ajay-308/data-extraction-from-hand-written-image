import os
import json
import tempfile
from pathlib import Path
from datetime import datetime
import streamlit as st

from medical_ocr_pipeline import MedicalDocumentPipeline, VectorDBManager

# ------------------ PAGE CONFIG ------------------
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

# ------------------ INIT ------------------
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
Upload PDFs or images — files will be processed and a **PROVER JSON** will be automatically downloaded.  
The full JSON is also saved in the output folder.
"""
)

if not st.session_state.pipeline:
    st.warning("Please initialize the pipeline from the sidebar to proceed.")
    st.stop()

# ------------------ SINGLE TAB: UPLOAD + PROCESS ------------------
uploaded = st.file_uploader(
    "📤 Upload PDF / Image file(s)",
    type=["pdf", "png", "jpg", "jpeg", "tiff"],
    accept_multiple_files=True,
)

if uploaded and st.button("🔄 Process files"):
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
            # Run pipeline
            res = st.session_state.pipeline.process_document(tmp.name, output_dir=output_dir)
            prover_json = res.get("prover_json", {})
            out_path = res.get("output_path")

            summary = {
                "original_filename": uf.name,
                "saved_prover_path": out_path,
                "document_header": prover_json.get("document_header"),
                "provenance": prover_json.get("provenance"),
                "quality_metrics": prover_json.get("quality_metrics"),
                "processed_at": datetime.utcnow().isoformat(),
            }

            st.session_state.processed_docs.insert(0, summary)
            st.success(f"✅ Processed: {uf.name}")

            # ✅ Auto-download JSON via JavaScript
            if out_path and os.path.exists(out_path):
                st.write(f"📁 Saved PROVER JSON: `{out_path}`")

                with open(out_path, "r", encoding="utf-8") as f:
                    json_text = f.read()

                # Escape backticks to prevent JS template literal errors
                safe_json_text = json_text.replace("`", "\\`")

                download_script = f"""
                <script>
                    const dataStr = `{safe_json_text}`;
                    const blob = new Blob([dataStr], {{ type: "application/json" }});
                    const url = URL.createObjectURL(blob);
                    const link = document.createElement('a');
                    link.href = url;
                    link.download = "{Path(out_path).name}";
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                    URL.revokeObjectURL(url);
                </script>
                """
                st.markdown(download_script, unsafe_allow_html=True)
            else:
                st.warning("⚠️ No output file found.")

        except Exception as e:
            st.error(f"❌ Error processing {uf.name}: {e}")

        finally:
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)

        prog.progress(i / len(uploaded))

    status.text("✅ All files processed successfully.")
    st.balloons()

st.markdown("---")
st.caption("⚠️ PROVER JSON files are saved in the output folder and downloaded automatically after processing.")
