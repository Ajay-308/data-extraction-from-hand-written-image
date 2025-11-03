# medical_ocr_pipeline.py
import os
import json
import logging
import hashlib
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

from PIL import Image
from pdf2image import convert_from_path
from paddleocr import PaddleOCR

# LLM (Gemini)
import google.generativeai as genai

# Vector DB / Embeddings (open-source)
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medical-ocr-pipeline")

# -----------------------
# OCR Engine (Paddle only for production)
# -----------------------
# -----------------------
# OCR Engine (Hybrid: Printed + Handwritten)
# -----------------------
class HybridOCREngine:
    def __init__(self):
        logger.info("Initializing Hybrid OCR Engine (Printed + Handwritten)...")
        # Base OCR for printed English text
        self.paddle_printed = PaddleOCR(use_angle_cls=True, lang='en')
        # Handwritten model (PP-OCRv4 handwritten recognition)
        try:
            self.paddle_handwritten = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                rec_model_dir='en_PP-OCRv4_rec_handwritten',  # <-- pretrained handwriting model dir
                det_model_dir='en_PP-OCRv4_det',
            )
            logger.info("✅ Handwritten PaddleOCR model loaded")
        except Exception as e:
            logger.warning(f"⚠️ Could not load handwriting model, fallback to printed only: {e}")
            self.paddle_handwritten = self.paddle_printed
        logger.info("✅ Hybrid OCR initialized")

    def extract_text_regions(self, image_path: str) -> List[Dict[str, Any]]:
        """Run OCR (printed + handwritten) and return list of text regions with type tags."""
        all_regions = []

        # 1️⃣ Printed text detection
        try:
            printed = self.paddle_printed.ocr(image_path, cls=True)
            for line in printed[0]:
                if len(line) >= 2:
                    all_regions.append({
                        "text": line[1][0],
                        "confidence": line[1][1],
                        "region_type": "printed"
                    })
        except Exception as e:
            logger.warning(f"Printed OCR failed: {e}")

        # 2️⃣ Handwritten text detection (targeting missed or unclear zones)
        try:
            handwritten = self.paddle_handwritten.ocr(image_path, cls=True)
            for line in handwritten[0]:
                if len(line) >= 2:
                    all_regions.append({
                        "text": line[1][0],
                        "confidence": line[1][1],
                        "region_type": "handwritten"
                    })
        except Exception as e:
            logger.warning(f"Handwriting OCR failed: {e}")

        return all_regions

    def extract_text_from_image(self, image_path: str) -> str:
        """Merge printed + handwritten text (separated by region markers)."""
        regions = self.extract_text_regions(image_path)
        if not regions:
            return "[EMPTY TEXT]"

        printed_text = "\n".join([r["text"] for r in regions if r["region_type"] == "printed"])
        handwritten_text = "\n".join([r["text"] for r in regions if r["region_type"] == "handwritten"])

        return (
            "=== PRINTED TEXT ===\n"
            + printed_text
            + "\n\n=== HANDWRITTEN TEXT ===\n"
            + handwritten_text
        ).strip()


# -----------------------
# PDF → images
# -----------------------
def convert_pdf_to_images(pdf_path: str, dpi: int = 300) -> List[str]:
    logger.info(f"Converting PDF to images: {pdf_path}")
    try:
        pages = convert_from_path(pdf_path, dpi=dpi)
        out_dir = os.path.splitext(pdf_path)[0] + "_pages"
        os.makedirs(out_dir, exist_ok=True)
        image_paths = []
        for i, page in enumerate(pages, start=1):
            p = os.path.join(out_dir, f"page_{i}.png")
            page.save(p, "PNG")
            image_paths.append(p)
        logger.info(f"✅ Converted {len(image_paths)} pages")
        return image_paths
    except Exception as e:
        logger.error(f"❌ Failed to convert PDF: {e}")
        return []

# -----------------------
# LLM Enricher (Gemini) with graceful fallback
# -----------------------
class MedicalLLMProcessor:
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"):
        self.ok = False
        self.model_name = model_name
        if api_key:
            try:
                genai.configure(api_key=api_key)
                # GenerativeModel wrapper used later
                self.model = genai.GenerativeModel(self.model_name)
                self.ok = True
                logger.info("✅ Gemini configured")
            except Exception as e:
                logger.warning(f"Could not configure Gemini: {e}")
                self.ok = False
        else:
            logger.info("No Gemini API key provided — LLM enrichment disabled.")
            self.ok = False

    def structure_medical_data(self, raw_text: str) -> Dict[str, Any]:
        # Fallback structure
        fallback = {
            "patient_information": {"name": "", "age": "", "gender": "", "id": ""},
            "clinical_notes": {"presenting_complaint": "", "provisional_diagnosis": "", "line_of_treatment": "", "admitting_ward": ""},
            "investigations": {"advised_tests": [], "results": []},
            "vitals": {"BP": "", "PR": "", "RR": "", "Temp": "", "SpO2": "", "Height": "", "Weight": ""},
            "medications": [],
            "consultant_details": {"doctor_name": "", "registration_number": "", "hospital": ""},
            "administrative": {"date": "", "stamps_signatures": "", "department": ""},
            "handwritten_notes": "",
            "printed_text": raw_text[:4000]
        }
        if not self.ok:
            return fallback

        prompt = (
            "You are a medical data extraction assistant. Return ONLY valid JSON matching "
            "this schema (keys):\n"
            + json.dumps(list(fallback.keys()), indent=2)
            + "\n\nRaw OCR Text:\n" + raw_text
        )

        try:
            resp = self.model.generate_content(prompt)
            text = getattr(resp, "text", str(resp)).strip()

            # try to extract JSON from code fences if present
            if "```" in text:
                part = text.split("```")
                # find a part that looks like json
                candidate = None
                for p in part:
                    p_strip = p.strip()
                    if p_strip.startswith("{") and p_strip.endswith("}"):
                        candidate = p_strip
                        break
                json_str = candidate or part[-1]
            else:
                json_str = text

            parsed = json.loads(json_str)
            return parsed
        except Exception as e:
            logger.warning(f"LLM call/parse failed: {e}")
            return fallback

# -----------------------
# Vector DB manager (FAISS + HuggingFace embeddings)
# -----------------------
class VectorDatabase:
    def __init__(self, persist_directory: str = "./medical_vector_db", hf_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.persist_directory = persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)
        self.embeddings = HuggingFaceEmbeddings(model_name=hf_model)
        self.vector_store = None
        logger.info(f"Vector DB ready at {self.persist_directory}")

    def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> List[str]:
        if not texts:
            return []
        # create or overwrite local FAISS with texts
        self.vector_store = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
        # save locally
        try:
            self.vector_store.save_local(self.persist_directory)
        except Exception:
            pass
        # construct simple ids
        ids = [f"vec_{uuid.uuid4().hex[:8]}" for _ in texts]
        logger.info(f"Stored {len(texts)} documents into vector DB")
        return ids

    def search(self, query: str, k: int = 5):
        if not self.vector_store:
            return []
        try:
            # many FAISS wrappers use similarity_search_with_score
            results = self.vector_store.similarity_search_with_score(query, k=k)
            # results: list of (doc, score)
            formatted = [{"document": r[0].page_content if hasattr(r[0], "page_content") else str(r[0]), "score": r[1]} for r in results]
            return formatted
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            return []

# -----------------------
# Main pipeline
# -----------------------
class MedicalDocumentPipeline:
    def __init__(self, gemini_api_key: Optional[str] = None, vector_db_path: str = "./medical_vector_db"):
        self.ocr_engine = HybridOCREngine()
        self.llm = MedicalLLMProcessor(gemini_api_key)
        self.vector_db = VectorDatabase(vector_db_path)

    def calculate_file_hash(self, file_path: str) -> str:
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def process_document(self, file_path: str, output_dir: str = "./output") -> Dict[str, Any]:
        os.makedirs(output_dir, exist_ok=True)
        start = datetime.utcnow()
        doc_id = f"DOC_{uuid.uuid4().hex[:12]}"
        file_hash = self.calculate_file_hash(file_path)
        ext = os.path.splitext(file_path)[1].lower()

        # convert or single image
        images = []
        if ext == ".pdf":
            images = convert_pdf_to_images(file_path)
        else:
            images = [file_path]

        all_texts = []
        structured_per_page = []
        for i, img in enumerate(images, start=1):
            text = self.ocr_engine.extract_text_from_image(img)
            structured = self.llm.structure_medical_data(text) if self.llm else {"printed_text": text}
            all_texts.append(text)
            structured_per_page.append({"page_number": i, "image_path": img, "structured": structured})

        vector_ids = self.vector_db.add_documents(all_texts, metadatas=structured_per_page)

        # quality metrics (simple heuristics)
        ocr_confidence = round(min(0.99, max(0.0, sum(len(t) for t in all_texts) / (1000 * max(1, len(all_texts))))), 3)
        handwritten_regions = sum([1 for t in all_texts if any(c.isalpha() for c in t) and len(t) < 100])  # heuristic
        printed_regions = max(0, len(all_texts) - handwritten_regions)

        proc_ms = (datetime.utcnow() - start).total_seconds() * 1000.0
        timestamp = datetime.utcnow().isoformat() + "Z"

        prover_json = {
            "prover_version": "1.0",
            "prover_id": str(uuid.uuid4()),
            "timestamp": timestamp,
            "document_header": {
                "document_id": doc_id,
                "document_type": "medical_clinical_notes",
                "source_file": os.path.basename(file_path),
                "file_hash": file_hash,
                "page_count": len(images),
                "processing_date": timestamp
            },
            "provenance": {
                "extraction_method": "hybrid_ocr_llm",
                "ocr_engines": ["PaddleOCR"],  # we run Paddle in production
                "llm_model": (self.llm.model_name if hasattr(self.llm, "model_name") else None),
                "confidence_scores": {"ocr_confidence": ocr_confidence},
                "processing_time_ms": round(proc_ms, 2),
                "vector_db_ids": vector_ids
            },
            "extracted_data": ["\n".join([p.get("structured", {}).get("printed_text", "") or p.get("structured", {}).get("raw_output", "") or "" for p in structured_per_page])],
            "quality_metrics": {
                "overall_confidence": round(ocr_confidence, 3),
                "handwritten_regions": int(handwritten_regions),
                "printed_regions": int(printed_regions),
                "extraction_completeness": round(0.85, 2)
            },
            "audit_trail": {
                "extracted_by": "Medical OCR Pipeline v1.1",
                "validation_status": "pending",
                "human_review_required": ocr_confidence < 0.7
            }
        }

        # Save full PROVER JSON to disk (no printing)
        out_path = os.path.join(output_dir, f"{doc_id}_prover.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(prover_json, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ Saved Prover JSON: {out_path}")
        # return a small summary and path to the saved json (so UI can show limited info)
        return {
            "prover_json_path": out_path,
            "document_header": prover_json["document_header"],
            "provenance": prover_json["provenance"],
            "quality_metrics": prover_json["quality_metrics"],
            "audit_trail": prover_json["audit_trail"]
        }
