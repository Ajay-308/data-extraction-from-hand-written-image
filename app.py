import streamlit as st
import os
import json
import sys
from pathlib import Path
from datetime import datetime
import tempfile
import shutil



# Import your pipeline classes (adjust import based on your file structure)
from medical_ocr_pipeline import (
    MedicalDocumentPipeline,
    VectorDatabase
)


# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Medical Document OCR",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# Custom CSS
# -------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .stExpander {
        border: 1px solid #ddd;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------
# Session State Init
# -------------------------
if 'processed_docs' not in st.session_state:
    st.session_state.processed_docs = []
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'vector_db_initialized' not in st.session_state:
    st.session_state.vector_db_initialized = False

# -------------------------
# Sidebar Configuration
# -------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/medical-doctor.png", width=80)
    st.title("⚙️ Configuration")
    
    # API Key input
    st.subheader("🔑 Gemini API Key")
    gemini_key = st.text_input(
        "Enter your Gemini API Key",
        type="password",
        help="Optional: Leave blank to skip LLM enrichment",
        value=""
    )
    
    # Vector DB settings
    st.subheader("💾 Vector Database")
    vector_db_path = st.text_input(
        "Vector DB Path",
        value="./medical_vector_db",
        help="Local directory for ChromaDB storage"
    )
    
    # Output settings
    st.subheader("📁 Output Settings")
    output_dir = st.text_input(
        "Output Directory",
        value="./output",
        help="Where to save Prover JSON files"
    )
    
    # Initialize pipeline button
    if st.button("🚀 Initialize Pipeline", type="primary"):
        with st.spinner("Initializing pipeline components..."):
            try:
                api_key = gemini_key if gemini_key.strip() else None
                st.session_state.pipeline = MedicalDocumentPipeline(
                    gemini_api_key=api_key,
                    vector_db_path=vector_db_path
                )
                st.session_state.vector_db_initialized = True
                st.success("✅ Pipeline initialized successfully!")
            except Exception as e:
                st.error(f"❌ Initialization failed: {str(e)}")
    
    # Stats
    st.divider()
    st.subheader("📊 Statistics")
    st.metric("Processed Documents", len(st.session_state.processed_docs))
    st.metric("Vector DB Status", 
              "✅ Ready" if st.session_state.vector_db_initialized else "⏸️ Not Initialized")

# -------------------------
# Main Content
# -------------------------
st.markdown('<p class="main-header">🏥 Medical Document OCR Pipeline</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Extract, Structure & Store Medical Documents with AI</p>', unsafe_allow_html=True)

# Check if pipeline is initialized
if not st.session_state.pipeline:
    st.warning("⚠️ Please initialize the pipeline from the sidebar before uploading documents.")
    st.info("""
    ### 📋 Getting Started:
    1. Enter your Gemini API Key (optional, for AI enrichment)
    2. Configure Vector DB and Output paths
    3. Click "Initialize Pipeline"
    4. Upload your medical documents
    """)
else:
    # Tabs for different functionalities
    tab1, tab2, tab3, tab4 = st.tabs([
        "📤 Upload & Process", 
        "🔍 Search Documents", 
        "📊 View Results",
        "💾 Export Data"
    ])
    
    # -------------------------
    # TAB 1: Upload & Process
    # -------------------------
    with tab1:
        st.subheader("Upload Medical Documents")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_files = st.file_uploader(
                "Choose PDF or Image files",
                type=["pdf", "png", "jpg", "jpeg", "tiff"],
                accept_multiple_files=True,
                help="Supports scanned PDFs, handwritten notes, and printed documents"
            )
        
        with col2:
            st.info("""
            **Supported Formats:**
            - PDF (single/multi-page)
            - PNG, JPG, JPEG
            - TIFF
            
            **Content Types:**
            - Printed text
            - Handwritten notes
            - Forms & tables
            - Signatures & stamps
            """)
        
        if uploaded_files:
            st.write(f"📁 **{len(uploaded_files)} file(s) selected**")
            
            if st.button("🔄 Process Documents", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {idx+1}/{len(uploaded_files)}: {uploaded_file.name}")
                    
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                        tmp_file.write(uploaded_file.getbuffer())
                        tmp_path = tmp_file.name
                    
                    try:
                        # Process document
                        result = st.session_state.pipeline.process_document(
                            tmp_path,
                            output_dir=output_dir
                        )
                        
                        # Store result
                        result['original_filename'] = uploaded_file.name
                        result['processed_at'] = datetime.now().isoformat()
                        st.session_state.processed_docs.append(result)
                        
                        # Success message
                        st.success(f"✅ Processed: {uploaded_file.name}")
                        
                    except Exception as e:
                        st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                    
                    finally:
                        # Clean up temp file
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
                    
                    # Update progress
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                status_text.text("✅ All documents processed!")
                st.balloons()
    
    # -------------------------
    # TAB 2: Search Documents
    # -------------------------
    with tab2:
        st.subheader("🔍 Semantic Search")
        
        search_query = st.text_input(
            "Enter search query",
            placeholder="e.g., 'patient with diabetes' or 'blood pressure readings'",
            help="Search across all processed documents using semantic similarity"
        )
        
        n_results = st.slider("Number of results", min_value=1, max_value=10, value=5)
        
        if st.button("🔎 Search", type="primary"):
            if search_query.strip():
                with st.spinner("Searching..."):
                    try:
                        results = st.session_state.pipeline.vector_db.search(
                            search_query,
                            n_results=n_results
                        )
                        
                        if results and 'documents' in results and results['documents']:
                            st.success(f"Found {len(results['documents'][0])} results")
                            
                            for idx, (doc, metadata, distance) in enumerate(zip(
                                results.get('documents', [[]])[0],
                                results.get('metadatas', [[]])[0],
                                results.get('distances', [[]])[0]
                            )):
                                with st.expander(f"Result {idx+1} - Similarity: {1-distance:.2%}"):
                                    st.write("**Document:**", doc[:500] + "..." if len(doc) > 500 else doc)
                                    st.json(metadata)
                        else:
                            st.warning("No results found. Try a different query.")
                    
                    except Exception as e:
                        st.error(f"Search failed: {str(e)}")
            else:
                st.warning("Please enter a search query")
    
    # -------------------------
    # TAB 3: View Results
    # -------------------------
    with tab3:
        st.subheader("📊 Processed Documents")
        
        if st.session_state.processed_docs:
            # Document selector
            doc_names = [doc.get('original_filename', f"Doc {i+1}") 
                        for i, doc in enumerate(st.session_state.processed_docs)]
            selected_doc_name = st.selectbox("Select document to view", doc_names)
            selected_idx = doc_names.index(selected_doc_name)
            selected_doc = st.session_state.processed_docs[selected_idx]
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                confidence = selected_doc.get('quality_metrics', {}).get('overall_confidence', 0) * 100
                st.metric("Confidence", f"{confidence:.1f}%")
            with col2:
                page_count = selected_doc.get('document_header', {}).get('page_count', 0)
                st.metric("Pages", page_count)
            with col3:
                proc_time = selected_doc.get('provenance', {}).get('processing_time_ms', 0)
                st.metric("Processing Time", f"{proc_time:.0f}ms")
            with col4:
                review_needed = selected_doc.get('audit_trail', {}).get('human_review_required', False)
                st.metric("Review Required", "⚠️ Yes" if review_needed else "✅ No")
            
            # Extracted data tabs
            data_tab1, data_tab2, data_tab3 = st.tabs([
                "📋 Structured Data",
                "📄 Full JSON",
                "🔬 Quality Metrics"
            ])
            
            with data_tab1:
                extracted = selected_doc.get('extracted_data', {})
                
                # Patient Info
                if 'patient_information' in extracted:
                    st.subheader("👤 Patient Information")
                    st.json(extracted['patient_information'])
                
                # Clinical Notes
                if 'clinical_notes' in extracted:
                    st.subheader("🩺 Clinical Notes")
                    st.json(extracted['clinical_notes'])
                
                # Vitals
                if 'vitals' in extracted:
                    st.subheader("❤️ Vital Signs")
                    vitals = extracted['vitals']
                    vcol1, vcol2, vcol3 = st.columns(3)
                    with vcol1:
                        st.metric("Blood Pressure", vitals.get('BP', 'N/A'))
                        st.metric("Pulse Rate", vitals.get('PR', 'N/A'))
                    with vcol2:
                        st.metric("Respiratory Rate", vitals.get('RR', 'N/A'))
                        st.metric("Temperature", vitals.get('Temp', 'N/A'))
                    with vcol3:
                        st.metric("SpO2", vitals.get('SpO2', 'N/A'))
                        st.metric("Weight", vitals.get('Weight', 'N/A'))
                
                # Medications
                if 'medications' in extracted and extracted['medications']:
                    st.subheader("💊 Medications")
                    for med in extracted['medications']:
                        st.write(f"- {med}")
            
            with data_tab2:
                st.json(selected_doc)
                
                # Download button
                json_str = json.dumps(selected_doc, indent=2, ensure_ascii=False)
                st.download_button(
                    label="📥 Download JSON",
                    data=json_str,
                    file_name=f"{selected_doc.get('document_header', {}).get('document_id', 'document')}_prover.json",
                    mime="application/json"
                )
            
            with data_tab3:
                quality = selected_doc.get('quality_metrics', {})
                st.metric("Overall Confidence", f"{quality.get('overall_confidence', 0)*100:.1f}%")
                st.metric("Handwritten Regions", quality.get('handwritten_regions', 0))
                st.metric("Printed Regions", quality.get('printed_regions', 0))
                st.metric("Extraction Completeness", f"{quality.get('extraction_completeness', 0)*100:.1f}%")
                
                # Provenance info
                st.subheader("🔍 Provenance")
                provenance = selected_doc.get('provenance', {})
                st.write(f"**Extraction Method:** {provenance.get('extraction_method', 'N/A')}")
                st.write(f"**OCR Engines:** {', '.join(provenance.get('ocr_engines', []))}")
                st.write(f"**LLM Model:** {provenance.get('llm_model', 'N/A')}")
        else:
            st.info("No documents processed yet. Go to 'Upload & Process' tab to get started.")
    
    # -------------------------
    # TAB 4: Export Data
    # -------------------------
    with tab4:
        st.subheader("💾 Export Options")
        
        if st.session_state.processed_docs:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### Export All Documents")
                if st.button("📦 Export as Bulk JSON"):
                    bulk_data = {
                        'export_date': datetime.now().isoformat(),
                        'total_documents': len(st.session_state.processed_docs),
                        'documents': st.session_state.processed_docs
                    }
                    json_str = json.dumps(bulk_data, indent=2, ensure_ascii=False)
                    st.download_button(
                        label="📥 Download Bulk Export",
                        data=json_str,
                        file_name=f"bulk_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            
            with col2:
                st.write("### Clear Session Data")
                if st.button("🗑️ Clear All Processed Documents", type="secondary"):
                    st.session_state.processed_docs = []
                    st.success("✅ Session data cleared")
                    st.rerun()
        else:
            st.info("No documents to export. Process some documents first.")

# -------------------------
# Footer
# -------------------------
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🏥 Medical Document OCR Pipeline v1.1 | Built with Streamlit, PaddleOCR & Gemini AI</p>
    <p>⚠️ For research and development purposes only. Always verify extracted data.</p>
</div>
""", unsafe_allow_html=True)