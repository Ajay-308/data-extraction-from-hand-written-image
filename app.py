import os
import json
import tempfile
from pathlib import Path
from datetime import datetime
import streamlit as st

from medical_ocr_pipeline import MedicalDocumentPipeline

# PAGE CONFIG 
st.set_page_config(page_title="Medical Document OCR", page_icon="🏥", layout="wide")

st.sidebar.image("https://img.icons8.com/color/96/000000/medical-doctor.png", width=80)
st.sidebar.title("⚙️ Configuration")

gemini_key = st.sidebar.text_input(
    "🔑 Gemini API Key (required)",
    type="password",
    help="Leave blank to skip LLM enrichment",
)
vector_db_path = st.sidebar.text_input("💾 Vector DB Path", value="./medical_vector_db")

# INIT pipeline
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

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
st.markdown("""
📤 **Upload PDFs or Images** — files will be processed automatically.  
Once processing is complete, a **PROVER JSON** file will be generated.  
You can **download it manually** using the **Download JSON** button below.  
*(No files are stored on the server.)*
""")


if not st.session_state.pipeline:
    st.warning("Please initialize the pipeline from the sidebar to proceed.")
    st.stop()

# UPLOAD + PROCESS 
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
            # 🚀 Run pipeline without writing to output folder
            res = st.session_state.pipeline.process_document(tmp.name)
            prover_json = res.get("prover_json", {})

            # Convert JSON to string (in-memory)
            json_str = json.dumps(prover_json, indent=2, ensure_ascii=False)
            file_name = f"{Path(uf.name).stem}_prover.json"

            st.success(f"✅ Processed and downloaded: {uf.name}")

            # ✅ Reliable browser download trigger (production-safe)
            st.download_button(
                label=f"⬇️ Download {file_name}",
                data=json_str.encode("utf-8"),
                file_name=file_name,
                mime="application/json"
            )

        except Exception as e:
            st.error(f"❌ Error processing {uf.name}: {e}")

        finally:
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)

        prog.progress(i / len(uploaded))

    status.text("✅ All files processed successfully.")
    st.balloons()

st.markdown("---")
st.caption("📥 Processed PROVER JSON files are automatically downloaded to your browser's default Downloads folder.")
