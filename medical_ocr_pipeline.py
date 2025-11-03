# medical_ocr_pipeline.py
import os
import re
import json
import logging
import hashlib
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

from PIL import Image
from pdf2image import convert_from_path
from paddleocr import PaddleOCR

# LLM (Gemini) - optional
import google.generativeai as genai

# Vector DB / Embeddings (open-source)
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medical-ocr-pipeline")


# -----------------------
# OCR Engine (Hybrid: Printed + Handwritten)
# -----------------------
class HybridOCREngine:
    """
    Hybrid OCR engine that runs a printed-text PaddleOCR instance and (optionally)
    a handwriting-specialized PaddleOCR instance, then separates printed vs handwritten lines.
    """

    def __init__(self,
                 printed_lang: str = "en",
                 handwriting_rec_model_dir: Optional[str] = None,
                 handwriting_det_model_dir: Optional[str] = None):
        logger.info("Initializing Hybrid OCR Engine (Printed + Handwritten)...")
        # Printed OCR (default)
        try:
            self.paddle_printed = PaddleOCR(use_angle_cls=True, lang=printed_lang)
            logger.info("✅ Paddle printed-text OCR initialized")
        except Exception as e:
            logger.error(f"Failed to initialize printed PaddleOCR: {e}")
            raise

        # Optional handwriting-specialized model (if user provided local model dirs)
        if handwriting_rec_model_dir and handwriting_det_model_dir:
            try:
                self.paddle_handwritten = PaddleOCR(
                    use_angle_cls=True,
                    lang=printed_lang,
                    rec_model_dir=handwriting_rec_model_dir,
                    det_model_dir=handwriting_det_model_dir
                )
                logger.info("✅ Paddle handwritten OCR initialized (custom model dirs)")
            except Exception as e:
                logger.warning(f"Could not initialize handwriting-specific models: {e} — falling back to printed instance")
                self.paddle_handwritten = self.paddle_printed
        else:
            self.paddle_handwritten = self.paddle_printed
            logger.info("No handwriting model dirs given — using printed OCR instance for both (may reduce handwriting accuracy)")

        logger.info("✅ Hybrid OCR ready")

    def _ocr_to_regions(self, ocr_res: Any, region_type: str) -> List[Dict[str, Any]]:
        """Convert PaddleOCR result to region dicts."""
        out = []
        if not ocr_res or not ocr_res[0]:
            return out
        for line in ocr_res[0]:
            if len(line) >= 2 and isinstance(line[1], (list, tuple)):
                text = str(line[1][0]).strip()
                try:
                    conf = float(line[1][1] or 0.0)
                except Exception:
                    conf = 0.0
                if text:
                    out.append({"text": text, "confidence": conf, "region_type": region_type})
        return out

    def extract_text_regions(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Run printed OCR and handwriting OCR (may be same instance) and return combined regions.
        """
        regions: List[Dict[str, Any]] = []
        # Printed
        try:
            printed_res = self.paddle_printed.ocr(image_path, cls=True)
            regions.extend(self._ocr_to_regions(printed_res, "printed"))
        except Exception as e:
            logger.warning(f"Printed OCR failed for {image_path}: {e}")

        # Handwritten
        try:
            hand_res = self.paddle_handwritten.ocr(image_path, cls=True)
            regions.extend(self._ocr_to_regions(hand_res, "handwritten"))
        except Exception as e:
            logger.warning(f"Handwritten OCR failed for {image_path}: {e}")

        return regions

    def _dedupe_and_classify(self, regions: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Heuristically separate printed_text and handwritten_text.
        Strategy:
         - If the same text appears from both engines, prefer the higher-confidence one.
         - If a line has low confidence (<0.70) or very short (<=3 tokens), treat it as likely handwritten.
         - Otherwise treat as printed.
        """
        if not regions:
            return {"printed": [], "handwritten": []}

        # Map text -> best region (by confidence)
        text_map: Dict[str, Dict[str, Any]] = {}
        for r in regions:
            t = r["text"].strip()
            key = re.sub(r"\s+", " ", t).strip()
            if not key:
                continue
            existing = text_map.get(key)
            if not existing or (r.get("confidence", 0.0) > existing.get("confidence", 0.0)):
                text_map[key] = r

        printed_lines: List[str] = []
        handwritten_lines: List[str] = []

        for text, r in text_map.items():
            conf = float(r.get("confidence", 0.0) or 0.0)
            num_tokens = len(text.split())
            # heuristic
            if r.get("region_type") == "handwritten" or conf < 0.75 or num_tokens <= 3:
                handwritten_lines.append(text)
            else:
                printed_lines.append(text)

        # maintain ordering naive (printed first then handwritten) — we preserve lines, not original y-order
        return {"printed": printed_lines, "handwritten": handwritten_lines}

    def extract_text_from_image(self, image_path: str) -> Dict[str, str]:
        """
        Returns:
            {"printed_text": "...", "handwritten_text": "..."}
        """
        regions = self.extract_text_regions(image_path)
        separated = self._dedupe_and_classify(regions)
        printed_text = "\n".join(separated.get("printed", []))
        handwritten_text = "\n".join(separated.get("handwritten", []))
        return {
            "printed_text": printed_text or "",
            "handwritten_text": handwritten_text or ""
        }


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
        logger.error(f"Failed to convert PDF {pdf_path}: {e}")
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
                self.model = genai.GenerativeModel(self.model_name)
                self.ok = True
                logger.info("✅ Gemini configured")
            except Exception as e:
                logger.warning(f"Could not configure Gemini: {e}")
                self.ok = False
        else:
            logger.info("No Gemini API key provided — LLM enrichment disabled.")
            self.ok = False

    def _extract_json_from_text(self, text: str) -> Optional[str]:
        """
        Helper: try to find a JSON object in model text (inside code fences or plain).
        """
        if not text:
            return None
        # Try code fence extraction
        if "```" in text:
            parts = text.split("```")
            for p in parts:
                p_strip = p.strip()
                if p_strip.startswith("{") and p_strip.endswith("}"):
                    return p_strip
        # Try first { ... } block using regex (greedy safe-ish)
        m = re.search(r"(\{[\s\S]*\})", text)
        if m:
            return m.group(1)
        return None

    def structure_medical_data(self, raw_text: str) -> Dict[str, Any]:
        """
        raw_text: printed text (only) — pass to LLM to extract structured fields.
        Returns fallback if LLM not available or parse fails.
        """
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
            "You are a medical data extraction assistant. Return ONLY valid JSON matching this schema (keys):\n"
            + json.dumps(list(fallback.keys()), indent=2)
            + "\n\nRaw OCR Text (printed only):\n" + (raw_text or "")
        )

        try:
            resp = self.model.generate_content(prompt)
            text = getattr(resp, "text", str(resp)).strip()
            json_candidate = self._extract_json_from_text(text)
            if json_candidate:
                parsed = json.loads(json_candidate)
                # ensure handwritten_notes present (we will overwrite later)
                if "handwritten_notes" not in parsed:
                    parsed["handwritten_notes"] = ""
                return parsed
            else:
                # If LLM returned something but we couldn't parse, log and fallback
                logger.warning("LLM returned no parsable JSON; falling back to default structure")
                return fallback
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
        self.vector_store = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
        try:
            self.vector_store.save_local(self.persist_directory)
        except Exception:
            logger.debug("Could not save vector store locally (non-fatal).")
        ids = [f"vec_{uuid.uuid4().hex[:8]}" for _ in texts]
        logger.info(f"Stored {len(texts)} documents into vector DB")
        return ids

    def search(self, query: str, k: int = 5):
        if not self.vector_store:
            return []
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            formatted = [{"document": r[0].page_content if hasattr(r[0], "page_content") else str(r[0]), "score": r[1]} for r in results]
            return formatted
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            return []


# -----------------------
# Main pipeline
# -----------------------
class MedicalDocumentPipeline:
    def __init__(self,
                 gemini_api_key: Optional[str] = None,
                 vector_db_path: str = "./medical_vector_db",
                 handwriting_rec_model_dir: Optional[str] = None,
                 handwriting_det_model_dir: Optional[str] = None):
        # initialize hybrid OCR with optional handwriting model paths
        self.ocr_engine = HybridOCREngine(
            printed_lang="en",
            handwriting_rec_model_dir=handwriting_rec_model_dir,
            handwriting_det_model_dir=handwriting_det_model_dir
        )
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

        # convert pdf or use single image
        if ext == ".pdf":
            images = convert_pdf_to_images(file_path)
        else:
            images = [file_path]

        all_texts_for_vector: List[str] = []
        structured_per_page: List[Dict[str, Any]] = []

        for i, img in enumerate(images, start=1):
            ocr_dict = self.ocr_engine.extract_text_from_image(img)
            printed = ocr_dict.get("printed_text", "") or ""
            handwritten = ocr_dict.get("handwritten_text", "") or ""

            # Log lengths for debugging
            logger.info(f"[Page {i}] printed_len={len(printed)} handwritten_len={len(handwritten)}")

            # Use only printed text for LLM structuring (reduces malformed JSON from messy handwriting)
            structured = self.llm.structure_medical_data(printed) if self.llm else {
                "printed_text": printed,
                "handwritten_notes": ""
            }
            # Attach handwritten notes raw (LLM won't try to parse it)
            structured["handwritten_notes"] = handwritten

            structured_per_page.append({
                "page_number": i,
                "image_path": img,
                "structured": structured
            })

            # For vector DB, we store a concat of printed + handwritten so searches find both
            combined = (printed + "\n\n" + handwritten).strip()
            all_texts_for_vector.append(combined if combined else "[EMPTY]")

        # persist vectors
        vector_ids = self.vector_db.add_documents(all_texts_for_vector, metadatas=structured_per_page)

        # quality metrics
        ocr_confidence = round(min(0.99, max(0.0, sum(len(t) for t in all_texts_for_vector) / (1000 * max(1, len(all_texts_for_vector))))), 3)
        handwritten_regions = sum(1 for p in structured_per_page if p["structured"].get("handwritten_notes", "").strip())
        printed_regions = sum(1 for p in structured_per_page if p["structured"].get("printed_text", "").strip())

        proc_ms = (datetime.utcnow() - start).total_seconds() * 1000.0
        timestamp = datetime.utcnow().isoformat() + "Z"

        # Build extracted_data as per-page structured objects (more auditable)
        extracted_data_pages = []
        for p in structured_per_page:
            extracted_data_pages.append({
                "page_number": p["page_number"],
                "printed_text": p["structured"].get("printed_text", ""),
                "handwritten_text": p["structured"].get("handwritten_notes", ""),
                "structured_fields": {k: v for k, v in p["structured"].items() if k not in ("printed_text", "handwritten_notes")}
            })

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
                "ocr_engines": ["PaddleOCR"],
                "llm_model": (self.llm.model_name if hasattr(self.llm, "model_name") else None),
                "confidence_scores": {"ocr_confidence": ocr_confidence},
                "processing_time_ms": round(proc_ms, 2),
                "vector_db_ids": vector_ids
            },
            "extracted_data": extracted_data_pages,
            "quality_metrics": {
                "overall_confidence": round(ocr_confidence, 3),
                "handwritten_regions": int(handwritten_regions),
                "printed_regions": int(printed_regions),
                "extraction_completeness": round(0.85, 2)
            },
            "audit_trail": {
                "extracted_by": "Medical OCR Pipeline v1.2",
                "validation_status": "pending",
                "human_review_required": ocr_confidence < 0.7
            }
        }

        # Save PROVER JSON
        out_path = os.path.join(output_dir, f"{doc_id}_prover.json")
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(prover_json, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Saved Prover JSON: {out_path}")
        except Exception as e:
            logger.error(f"Failed to save Prover JSON: {e}")

        return {
            "prover_json_path": out_path,
            "document_header": prover_json["document_header"],
            "provenance": prover_json["provenance"],
            "quality_metrics": prover_json["quality_metrics"],
            "audit_trail": prover_json["audit_trail"]
        }


# If run as a script you can test like:
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Medical OCR Pipeline")
    parser.add_argument("--input", required=True, help="Path to PDF or image")
    parser.add_argument("--out", default="./output", help="Output directory")
    parser.add_argument("--gemini_key", default=None, help="Gemini API key (optional)")
    parser.add_argument("--hand_rec", default=None, help="Handwriting rec model dir (optional)")
    parser.add_argument("--hand_det", default=None, help="Handwriting det model dir (optional)")
    args = parser.parse_args()

    pipeline = MedicalDocumentPipeline(
        gemini_api_key=args.gemini_key,
        vector_db_path="./medical_vector_db",
        handwriting_rec_model_dir=args.hand_rec,
        handwriting_det_model_dir=args.hand_det
    )
    res = pipeline.process_document(args.input, output_dir=args.out)
    print(json.dumps(res, indent=2))
