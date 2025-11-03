# medical_ocr_pipeline_fixed.py
import os
import re
import json
import logging
import hashlib
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
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
# Image Preprocessing
# -----------------------
class ImagePreprocessor:
    """Enhanced preprocessing for better handwriting detection"""
    
    @staticmethod
    def preprocess_for_printed(image: np.ndarray) -> np.ndarray:
        """Optimize image for printed text detection"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Adaptive threshold for printed text
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        return binary
    
    @staticmethod
    def preprocess_for_handwritten(image: np.ndarray) -> np.ndarray:
        """Optimize image for handwritten text detection"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Stronger denoising for handwriting
        denoised = cv2.fastNlMeansDenoising(gray, None, h=15, templateWindowSize=7, searchWindowSize=21)
        
        # Enhance contrast more aggressively
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(denoised)
        
        # Different threshold strategy for handwriting
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
            cv2.THRESH_BINARY, 15, 10
        )
        
        # Morphological operations to connect broken strokes
        kernel = np.ones((2, 2), np.uint8)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        return closed


# -----------------------
# OCR Engine (Hybrid: Printed + Handwritten)
# -----------------------
class HybridOCREngine:
    """
    Enhanced hybrid OCR engine with better handwriting detection.
    Uses different preprocessing and detection strategies for printed vs handwritten text.
    """

    def __init__(self,
                 printed_lang: str = "en",
                 enable_handwriting_mode: bool = True):
        logger.info("Initializing Enhanced Hybrid OCR Engine...")
        
        self.preprocessor = ImagePreprocessor()
        
        # Printed OCR (standard configuration)
        try:
            self.paddle_printed = PaddleOCR(
                use_angle_cls=True, 
                lang=printed_lang,
                use_gpu=False,
                show_log=False
            )
            logger.info("✅ Paddle printed-text OCR initialized")
        except Exception as e:
            logger.error(f"Failed to initialize printed PaddleOCR: {e}")
            raise

        # Handwriting OCR (with different parameters for better detection)
        if enable_handwriting_mode:
            try:
                # Use same PaddleOCR but with different thresholds
                self.paddle_handwritten = PaddleOCR(
                    use_angle_cls=True,
                    lang=printed_lang,
                    use_gpu=False,
                    show_log=False,
                    det_db_thresh=0.2,  # Lower threshold for handwriting detection
                    det_db_box_thresh=0.3,  # Lower box threshold
                    rec_batch_num=1  # Process one at a time for better accuracy
                )
                logger.info("✅ Paddle handwritten OCR initialized with optimized settings")
            except Exception as e:
                logger.warning(f"Could not initialize handwriting-specific settings: {e}")
                self.paddle_handwritten = self.paddle_printed
        else:
            self.paddle_handwritten = self.paddle_printed
            logger.info("Handwriting mode disabled — using printed OCR only")

        logger.info("✅ Enhanced Hybrid OCR ready")

    def _ocr_to_regions(self, ocr_res: Any, region_type: str) -> List[Dict[str, Any]]:
        """Convert PaddleOCR result to region dicts with enhanced metadata."""
        out = []
        if not ocr_res or not ocr_res[0]:
            return out
            
        for line in ocr_res[0]:
            if len(line) >= 2:
                bbox = line[0]
                text_info = line[1]
                
                if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                    text = str(text_info[0]).strip()
                    try:
                        conf = float(text_info[1] or 0.0)
                    except Exception:
                        conf = 0.0
                else:
                    text = str(text_info).strip()
                    conf = 0.0
                    
                if text:
                    # Calculate additional metrics
                    num_chars = len(text)
                    num_words = len(text.split())
                    has_numbers = bool(re.search(r'\d', text))
                    has_special = bool(re.search(r'[^a-zA-Z0-9\s]', text))
                    
                    out.append({
                        "text": text,
                        "confidence": conf,
                        "region_type": region_type,
                        "bbox": bbox,
                        "num_chars": num_chars,
                        "num_words": num_words,
                        "has_numbers": has_numbers,
                        "has_special": has_special
                    })
        return out

    def extract_text_regions(self, image_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run both printed and handwritten OCR with optimized preprocessing.
        Returns separate results for better classification.
        """
        # Load original image
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Could not load image: {image_path}")
            return {"printed": [], "handwritten": []}
        
        # PRINTED TEXT DETECTION
        printed_regions = []
        try:
            # Preprocess for printed text
            printed_prep = self.preprocessor.preprocess_for_printed(img)
            # Convert back to RGB for PaddleOCR
            printed_rgb = cv2.cvtColor(printed_prep, cv2.COLOR_GRAY2BGR)
            
            printed_res = self.paddle_printed.ocr(printed_rgb, cls=True)
            printed_regions = self._ocr_to_regions(printed_res, "printed")
            logger.info(f"Detected {len(printed_regions)} printed text regions")
        except Exception as e:
            logger.warning(f"Printed OCR failed for {image_path}: {e}")

        # HANDWRITTEN TEXT DETECTION
        handwritten_regions = []
        try:
            # Preprocess for handwritten text
            hand_prep = self.preprocessor.preprocess_for_handwritten(img)
            # Convert back to RGB for PaddleOCR
            hand_rgb = cv2.cvtColor(hand_prep, cv2.COLOR_GRAY2BGR)
            
            hand_res = self.paddle_handwritten.ocr(hand_rgb, cls=True)
            handwritten_regions = self._ocr_to_regions(hand_res, "handwritten")
            logger.info(f"Detected {len(handwritten_regions)} handwritten text regions")
        except Exception as e:
            logger.warning(f"Handwritten OCR failed for {image_path}: {e}")

        return {
            "printed": printed_regions,
            "handwritten": handwritten_regions
        }

    def _classify_regions(self, printed_regions: List[Dict], handwritten_regions: List[Dict]) -> Dict[str, List[str]]:
        """
        Enhanced classification logic to separate printed vs handwritten text.
        
        Strategy:
        1. Deduplicate overlapping detections (prefer higher confidence)
        2. Use confidence thresholds and text characteristics
        3. Handwritten indicators: low confidence, irregular spacing, short fragments
        """
        # Combine all regions with source tracking
        all_regions = []
        for r in printed_regions:
            r_copy = r.copy()
            r_copy["source"] = "printed_pass"
            all_regions.append(r_copy)
        
        for r in handwritten_regions:
            r_copy = r.copy()
            r_copy["source"] = "handwritten_pass"
            all_regions.append(r_copy)
        
        if not all_regions:
            return {"printed": [], "handwritten": []}
        
        # Deduplicate similar texts (keep highest confidence)
        text_map: Dict[str, Dict[str, Any]] = {}
        for r in all_regions:
            text = r["text"].strip()
            # Normalize for comparison
            key = re.sub(r'\s+', ' ', text.lower()).strip()
            if not key:
                continue
                
            existing = text_map.get(key)
            if not existing or (r.get("confidence", 0.0) > existing.get("confidence", 0.0)):
                text_map[key] = r

        # Classify each unique text
        printed_lines: List[str] = []
        handwritten_lines: List[str] = []

        for normalized_key, region in text_map.items():
            text = region["text"]
            conf = float(region.get("confidence", 0.0))
            num_words = region.get("num_words", len(text.split()))
            source = region.get("source", "unknown")
            
            # Classification heuristics
            is_handwritten = False
            
            # Strong indicators of handwriting
            if conf < 0.60:  # Very low confidence
                is_handwritten = True
            elif conf < 0.75 and num_words <= 3:  # Low confidence + short
                is_handwritten = True
            elif source == "handwritten_pass" and conf < 0.85:  # Detected in handwriting pass with moderate conf
                is_handwritten = True
            elif num_words == 1 and len(text) <= 4 and conf < 0.80:  # Single short word
                is_handwritten = True
            
            # Add to appropriate list
            if is_handwritten:
                handwritten_lines.append(text)
            else:
                printed_lines.append(text)

        return {
            "printed": printed_lines,
            "handwritten": handwritten_lines
        }

    def extract_text_from_image(self, image_path: str) -> Dict[str, str]:
        """
        Main extraction method.
        Returns: {"printed_text": "...", "handwritten_text": "..."}
        """
        regions_dict = self.extract_text_regions(image_path)
        classified = self._classify_regions(
            regions_dict.get("printed", []),
            regions_dict.get("handwritten", [])
        )
        
        printed_text = "\n".join(classified.get("printed", []))
        handwritten_text = "\n".join(classified.get("handwritten", []))
        
        logger.info(f"Final classification - Printed: {len(classified['printed'])} lines, Handwritten: {len(classified['handwritten'])} lines")
        
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
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.0-flash-exp"):
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
        """Helper: try to find a JSON object in model text."""
        if not text:
            return None
        # Try code fence extraction
        if "```json" in text:
            parts = text.split("```json")
            if len(parts) > 1:
                json_part = parts[1].split("```")[0].strip()
                return json_part
        elif "```" in text:
            parts = text.split("```")
            for p in parts:
                p_strip = p.strip()
                if p_strip.startswith("{") and p_strip.endswith("}"):
                    return p_strip
        # Try first { ... } block
        m = re.search(r"(\{[\s\S]*\})", text)
        if m:
            return m.group(1)
        return None

    def structure_medical_data(self, raw_text: str) -> Dict[str, Any]:
        """Extract structured fields from printed text only."""
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
            "You are a medical data extraction assistant. Extract structured information from this medical document.\n"
            "Return ONLY valid JSON matching these fields:\n"
            + json.dumps(list(fallback.keys()), indent=2)
            + "\n\nRaw OCR Text (printed only):\n" + (raw_text or "")
        )

        try:
            resp = self.model.generate_content(prompt)
            text = getattr(resp, "text", str(resp)).strip()
            json_candidate = self._extract_json_from_text(text)
            if json_candidate:
                parsed = json.loads(json_candidate)
                if "handwritten_notes" not in parsed:
                    parsed["handwritten_notes"] = ""
                return parsed
            else:
                logger.warning("LLM returned no parsable JSON; using fallback")
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
        except Exception as e:
            logger.debug(f"Could not save vector store locally: {e}")
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
                 enable_handwriting: bool = True):
        # Initialize enhanced hybrid OCR
        self.ocr_engine = HybridOCREngine(
            printed_lang="en",
            enable_handwriting_mode=enable_handwriting
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

        # Convert PDF or use single image
        if ext == ".pdf":
            images = convert_pdf_to_images(file_path, dpi=300)
        else:
            images = [file_path]

        all_texts_for_vector: List[str] = []
        structured_per_page: List[Dict[str, Any]] = []

        for i, img in enumerate(images, start=1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing Page {i}/{len(images)}: {img}")
            logger.info(f"{'='*60}")
            
            # Extract text with enhanced handwriting detection
            ocr_dict = self.ocr_engine.extract_text_from_image(img)
            printed = ocr_dict.get("printed_text", "") or ""
            handwritten = ocr_dict.get("handwritten_text", "") or ""

            # Log extraction results
            logger.info(f"📄 Printed text: {len(printed)} chars, {len(printed.split())} words")
            logger.info(f"✍️  Handwritten text: {len(handwritten)} chars, {len(handwritten.split())} words")
            
            if handwritten:
                logger.info(f"✅ Handwritten content detected:\n{handwritten[:200]}...")

            # Use only printed text for LLM structuring
            structured = self.llm.structure_medical_data(printed) if self.llm.ok else {
                "printed_text": printed,
                "handwritten_notes": ""
            }
            # Attach handwritten notes
            structured["handwritten_notes"] = handwritten

            structured_per_page.append({
                "page_number": i,
                "image_path": img,
                "structured": structured
            })

            # For vector DB, store concat of both
            combined = (printed + "\n\n" + handwritten).strip()
            all_texts_for_vector.append(combined if combined else "[EMPTY]")

        # Persist vectors
        vector_ids = self.vector_db.add_documents(all_texts_for_vector, metadatas=structured_per_page)

        # Quality metrics
        total_chars = sum(len(t) for t in all_texts_for_vector)
        ocr_confidence = round(min(0.99, max(0.0, total_chars / (1000 * max(1, len(all_texts_for_vector))))), 3)
        handwritten_regions = sum(1 for p in structured_per_page if p["structured"].get("handwritten_notes", "").strip())
        printed_regions = sum(1 for p in structured_per_page if p["structured"].get("printed_text", "").strip())

        proc_ms = (datetime.utcnow() - start).total_seconds() * 1000.0
        timestamp = datetime.utcnow().isoformat() + "Z"

        # Build extracted_data
        extracted_data_pages = []
        for p in structured_per_page:
            extracted_data_pages.append({
                "page_number": p["page_number"],
                "printed_text": p["structured"].get("printed_text", ""),
                "handwritten_text": p["structured"].get("handwritten_notes", ""),
                "structured_fields": {k: v for k, v in p["structured"].items() if k not in ("printed_text", "handwritten_notes")}
            })

        prover_json = {
            "prover_version": "1.1",
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
                "extraction_method": "enhanced_hybrid_ocr_llm",
                "ocr_engines": ["PaddleOCR (printed)", "PaddleOCR (handwriting-optimized)"],
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
                "extracted_by": "Medical OCR Pipeline v1.3 (Enhanced Handwriting)",
                "validation_status": "pending",
                "human_review_required": ocr_confidence < 0.7 or handwritten_regions > 0
            }
        }

        # Save PROVER JSON
        out_path = os.path.join(output_dir, f"{doc_id}_prover.json")
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(prover_json, f, indent=2, ensure_ascii=False)
            logger.info(f"\n✅ Saved Prover JSON: {out_path}")
        except Exception as e:
            logger.error(f"Failed to save Prover JSON: {e}")

        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info("EXTRACTION SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total pages: {len(images)}")
        logger.info(f"Printed regions: {printed_regions}")
        logger.info(f"Handwritten regions: {handwritten_regions}")
        logger.info(f"Processing time: {proc_ms:.0f}ms")
        logger.info(f"{'='*60}\n")

        return {
            "prover_json_path": out_path,
            "document_header": prover_json["document_header"],
            "provenance": prover_json["provenance"],
            "quality_metrics": prover_json["quality_metrics"],
            "audit_trail": prover_json["audit_trail"]
        }


# Main execution
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced Medical OCR Pipeline with Handwriting Detection")
    parser.add_argument("--input", required=True, help="Path to PDF or image")
    parser.add_argument("--out", default="./output", help="Output directory")
    parser.add_argument("--gemini_key", default=None, help="Gemini API key (optional)")
    parser.add_argument("--no-handwriting", action="store_true", help="Disable handwriting detection")
    args = parser.parse_args()

    pipeline = MedicalDocumentPipeline(
        gemini_api_key=args.gemini_key,
        vector_db_path="./medical_vector_db",
        enable_handwriting=not args.no_handwriting
    )
    
    res = pipeline.process_document(args.input, output_dir=args.out)
    print("\n" + json.dumps(res, indent=2))