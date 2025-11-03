
import os
import json
import uuid
import hashlib
import logging
import easyocr
from datetime import datetime
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
from PIL import Image
import fitz 
from paddleocr import PaddleOCR
import pytesseract
from sentence_transformers import SentenceTransformer
import chromadb
import google.generativeai as genai
import re

pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medical-ocr-pipeline")



class ProverJSON:
    @staticmethod
    def create(document_info: Dict, extracted_data: Dict, metadata: Dict) -> Dict:
        prover_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat() + "Z"
        return {
            "prover_version": "1.0",
            "prover_id": prover_id,
            "timestamp": timestamp,
            "document_header": {
                "document_id": document_info.get("document_id"),
                "document_type": document_info.get("document_type", "medical_record"),
                "source_file": document_info.get("source_file"),
                "file_hash": document_info.get("file_hash"),
                "page_count": document_info.get("page_count", 1),
                "processing_date": timestamp
            },
            "provenance": {
                "extraction_method": metadata.get("extraction_method", "hybrid_ocr"),
                "ocr_engines": metadata.get("ocr_engines", []),
                "llm_model": metadata.get("llm_model"),
                "confidence_scores": metadata.get("confidence_scores", {}),
                "processing_time_ms": metadata.get("processing_time_ms"),
                "vector_db_ids": metadata.get("vector_db_ids", [])
            },
            "extracted_data": extracted_data,
            "text_analysis": {
                "handwritten_text": extracted_data.get("handwritten_notes", ""),
                "printed_text": extracted_data.get("printed_text", ""),
                "handwritten_items": [],
                "printed_items": []
            },

            "quality_metrics": {
                "overall_confidence": metadata.get("overall_confidence", 0.0),
                "handwritten_regions": metadata.get("handwritten_regions", 0),
                "printed_regions": metadata.get("printed_regions", 0),
                "extraction_completeness": metadata.get("completeness", 0.0)
            },
            "audit_trail": {
                "extracted_by": "Medical OCR Pipeline v1.2",
                "validation_status": "pending",
                "human_review_required": metadata.get("review_required", False)
            }
        }

# Preprocessor
class DocumentPreprocessor:
    @staticmethod
    def pdf_to_images(pdf_path: str, dpi: int = 300) -> List[np.ndarray]:
        images: List[np.ndarray] = []
        pdf_doc = fitz.open(pdf_path)
        for p in range(len(pdf_doc)):
            page = pdf_doc[p]
            mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
            pix = page.get_pixmap(matrix=mat)
            mode = "RGB" if pix.n >= 3 else "L"
            img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
            img_np = np.array(img.convert("RGB"))
            images.append(img_np)
        pdf_doc.close()
        return images

    @staticmethod
    def preprocess_image(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return cleaned

# Hybrid OCR Engine
class HybridOCREngine:
    """Uses PaddleOCR for printed text and Tesseract for handwritten + fallback."""
    def __init__(self, handwritten_threshold: float = 0.7):
        self.handwritten_threshold = handwritten_threshold
        self.available_engines: List[str] = []

        logger.info("Initializing PaddleOCR...")
        try:
            self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')
            self.available_engines.append("PaddleOCR")
            logger.info("✅ PaddleOCR initialized")
        except Exception as e:
            logger.warning(f"PaddleOCR init failed: {e}")
            self.paddle_ocr = None
        try:
            _ = pytesseract.get_tesseract_version()
            self.available_engines.append("Tesseract")
            logger.info("✅ Tesseract available")
        except Exception as e:
            logger.warning(f"Tesseract not available: {e}")

        try:
            self.easyocr_reader = easyocr.Reader(['en'])
            self.available_engines.append("EasyOCR")
            logger.info("✅ EasyOCR initialized for handwriting")
        except Exception as e:
            logger.warning(f"EasyOCR init failed: {e}")
            self.easyocr_reader = None
    def extract_text_easyocr(self, image: np.ndarray) -> Dict:
        """
        Extract text using EasyOCR (useful for handwritten or mixed text).
        Returns {items, raw_text, confidence}.
        """
        if self.easyocr_reader is None:
            logger.warning("EasyOCR not initialized; skipping.")
            return {"items": [], "raw_text": "", "confidence": 0.0}

        try:
            results = self.easyocr_reader.readtext(image)
        except Exception as e:
            logger.warning(f"EasyOCR failed: {e}")
            return {"items": [], "raw_text": "", "confidence": 0.0}

        extracted_items, all_text, total_conf, count = [], [], 0.0, 0
        for res in results:
            try:
                bbox, text, conf = res
                extracted_items.append({
                    "text": text,
                    "confidence": round(float(conf), 3),
                    "bbox": bbox,
                    "engine": "EasyOCR"
                })
                all_text.append(text)
                total_conf += conf
                count += 1
            except Exception:
                continue

        avg_conf = (total_conf / count) if count else 0.0
        return {"items": extracted_items, "raw_text": " ".join(all_text), "confidence": avg_conf}


    def extract_text_paddle(self, image: np.ndarray) -> Dict:
        if image is None:
            return {"items": [], "raw_text": "", "confidence": 0.0}
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        else:
            image_rgb = image
        try:
            result = self.paddle_ocr.ocr(image_rgb, cls=True)
        except Exception as e:
            logger.warning(f"PaddleOCR failed: {e}")
            return {"items": [], "raw_text": "", "confidence": 0.0}
        extracted_items, all_text, total_conf, count = [], [], 0.0, 0
        if result and isinstance(result, list):
            for block in result:
                for line in block or []:
                    try:
                        bbox = line[0]
                        text = str(line[1][0])
                        conf = float(line[1][1]) if len(line[1]) > 1 else 0.0
                        extracted_items.append({"text": text, "confidence": round(conf, 3), "bbox": bbox, "engine": "PaddleOCR"})
                        all_text.append(text)
                        total_conf += conf
                        count += 1
                    except Exception:
                        continue
        avg_conf = (total_conf / count) if count else 0.0
        return {"items": extracted_items, "raw_text": "\n".join(all_text), "confidence": avg_conf}

    def extract_text_tesseract(self, image: np.ndarray) -> Dict:
        if image is None:
            return {"items": [], "raw_text": "", "confidence": 0.0}
        img_for_tess = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image
        try:
            data = pytesseract.image_to_data(img_for_tess, output_type=pytesseract.Output.DICT)
        except Exception as e:
            logger.warning(f"Tesseract failed: {e}")
            return {"items": [], "raw_text": "", "confidence": 0.0}
        extracted_items, all_text, total_conf, count = [], [], 0.0, 0
        for i, txt in enumerate(data.get("text", [])):
            txt = (txt or "").strip()
            try:
                conf = float(data["conf"][i]) / 100.0
            except:
                conf = 0.0
            if txt:
                bbox = [data["left"][i], data["top"][i], data["left"][i]+data["width"][i], data["top"][i]+data["height"][i]]
                extracted_items.append({"text": txt, "confidence": round(conf, 3), "bbox": bbox, "engine": "Tesseract"})
                all_text.append(txt)
                total_conf += conf
                count += 1
        avg_conf = (total_conf / count) if count else 0.0
        return {"items": extracted_items, "raw_text": " ".join(all_text), "confidence": avg_conf}

    def hybrid_extract(self, image: np.ndarray) -> Dict:
        # Run each extractor only if initialized
        paddle_res = {"items": [], "raw_text": "", "confidence": 0.0}
        tess_res = {"items": [], "raw_text": "", "confidence": 0.0}
        easy_res = {"items": [], "raw_text": "", "confidence": 0.0}

        if getattr(self, "paddle_ocr", None) is not None:
            paddle_res = self.extract_text_paddle(image)
        if "Tesseract" in self.available_engines:
            tess_res = self.extract_text_tesseract(image)
        if getattr(self, "easyocr_reader", None) is not None:
            easy_res = self.extract_text_easyocr(image)

        # Combine text from engines (printed-first preference, but concat all)
        texts = []
        if paddle_res.get("raw_text"):
            texts.append(paddle_res["raw_text"])
        if tess_res.get("raw_text"):
            texts.append(tess_res["raw_text"])
        if easy_res.get("raw_text"):
            texts.append(easy_res["raw_text"])
        combined = "\n".join(t for t in texts if t).strip()

        all_items = paddle_res["items"] + tess_res["items"] + easy_res["items"]
        # safe avg: ignore NaNs
        confs = [float(x) for x in [
            paddle_res.get("confidence", 0.0),
            tess_res.get("confidence", 0.0),
            easy_res.get("confidence", 0.0)
        ]]
        avg_conf = float(np.mean(confs)) if confs else 0.0

        handwritten_items, printed_items, handwritten_text, printed_text = [], [], [], []
        for item in all_items:
            conf = item.get("confidence", 0.0)
            text = item.get("text", "")
            if conf < self.handwritten_threshold:
                handwritten_items.append(item)
                handwritten_text.append(text)
            else:
                printed_items.append(item)
                printed_text.append(text)

        return {
            "items": all_items,
            "raw_text": combined,
            "confidence": avg_conf,
            "engines_used": list(self.available_engines),
            "handwritten_items": handwritten_items,
            "printed_items": printed_items,
            "handwritten_text": " ".join(handwritten_text),
            "printed_text": " ".join(printed_text)
        }


# LLM Enricher (Gemini)
class LLMEnricher:
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key
        self.model_name = model_name
        self.ok = False
        if api_key:
            try:
                genai.configure(api_key=api_key)
                self.ok = True
                logger.info(f"✅ Gemini API configured with model: {model_name}")
            except Exception as e:
                logger.warning(f"Could not configure genai: {e}")
        else:
            logger.warning("No Gemini API key provided. LLM enrichment will be skipped.")

    def structure_medical_data(self, raw_text: str, handwritten_text: str = "", image: Optional[np.ndarray] = None) -> Dict:
        fallback = {
            "patient_information": {"name": "", "age": "", "gender": "", "id": ""},
            "clinical_notes": {
                "presenting_complaint": "", 
                "provisional_diagnosis": "", 
                "line_of_treatment": "", 
                "admitting_ward": ""
            },
            "investigations": {"advised_tests": [], "results": []},
            "vitals": {"BP": "", "PR": "", "RR": "", "Temp": "", "SpO2": "", "Height": "", "Weight": ""},
            "medications": [],
            "consultant_details": {"doctor_name": "", "registration_number": "", "hospital": ""},
            "administrative": {"date": "", "stamps_signatures": "", "department": ""},
            "handwritten_notes": handwritten_text,
            "printed_text": raw_text[:4000]
        }
        if not self.ok:
            logger.info("Skipping LLM enrichment - no API key configured")
            return fallback

        prompt = f"""You are a medical data extraction assistant. Extract structured information from the following medical document OCR text.

PRINTED TEXT:
{raw_text[:2500]}

HANDWRITTEN TEXT (if any):
{handwritten_text[:500]}

Return ONLY valid JSON matching this exact structure (no explanation):
{{
  "patient_information": {{"name": "", "age": "", "gender": "", "id": ""}},
  "clinical_notes": {{"presenting_complaint": "", "provisional_diagnosis": "", "line_of_treatment": "", "admitting_ward": ""}},
  "investigations": {{"advised_tests": [], "results": []}},
  "vitals": {{"BP": "", "PR": "", "RR": "", "Temp": "", "SpO2": "", "Height": "", "Weight": ""}},
  "medications": [],
  "consultant_details": {{"doctor_name": "", "registration_number": "", "hospital": ""}},
  "administrative": {{"date": "", "stamps_signatures": "", "department": ""}},
  "handwritten_notes": "{handwritten_text[:500]}",
  "printed_text": "{raw_text[:500]}"
}}"""

        try:
            model = genai.GenerativeModel(self.model_name)
            resp = model.generate_content(prompt)
            text = resp.text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            parsed = json.loads(text)
            logger.info("✅ LLM enrichment successful")
            return parsed
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            return fallback


# Vector DB Manager
class VectorDBManager:
    def __init__(self, persist_directory: str = "./chroma_db"):
        try:
            self.client = chromadb.PersistentClient(path=persist_directory)
            self.collection = self.client.get_or_create_collection(name="medical_documents")
        except Exception:
            self.client = chromadb.Client()
            self.collection = self.client.create_collection(name="medical_documents")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info(f"✅ Vector DB ready at {persist_directory}")

    def add_document(self, doc_id: str, text: str, metadata: Dict) -> str:
        if not text or len(text.strip()) == 0:
            logger.warning("Empty text, skipping vector DB insertion")
            return ""
        embedding = self.embedder.encode(text).tolist()
        vector_id = f"{doc_id}_{uuid.uuid4().hex[:8]}"
        try:
            self.collection.add(
                ids=[vector_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[metadata]
            )
            logger.info(f"✅ Added document to vector DB: {vector_id}")
        except Exception as e:
            logger.warning(f"Failed to add to vector DB: {e}")
            return ""
        return vector_id
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        try:
            query_embedding = self.embedder.encode(query).tolist()
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k
            )
            formatted_results = []
            if results and results.get("documents"):
                for i in range(len(results["documents"][0])):
                    formatted_results.append({
                        "document": results["documents"][0][i],
                        "score": 1 - results["distances"][0][i] if "distances" in results else 0,
                        "metadata": results["metadatas"][0][i] if "metadatas" in results else {}
                    })
            return formatted_results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

# Main Pipeline
class MedicalDocumentPipeline:
    def __init__(self, gemini_api_key: Optional[str] = None, vector_db_path: str = "./chroma_db"):
        logger.info("Initializing Medical Document Pipeline...")
        self.preprocessor = DocumentPreprocessor()
        self.ocr_engine = HybridOCREngine(handwritten_threshold=0.7)
        self.llm_enricher = LLMEnricher(api_key=gemini_api_key)
        self.vector_db = VectorDBManager(vector_db_path)
        logger.info("✅ Pipeline initialized successfully")

    def calculate_file_hash(self, file_path: str) -> str:
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def process_document(self, file_path: str, output_dir: str = "./output") -> Dict:
        logger.info(f"Processing document: {file_path}")
        start = datetime.now()
        os.makedirs(output_dir, exist_ok=True)
        doc_id = f"DOC_{uuid.uuid4().hex[:12]}"
        file_hash = self.calculate_file_hash(file_path)
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            logger.info("Converting PDF to images...")
            images = self.preprocessor.pdf_to_images(file_path)
        else:
            logger.info("Loading image file...")
            bgr = cv2.imread(file_path)
            if bgr is None:
                raise FileNotFoundError(f"Cannot read file: {file_path}")
            images = [cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)]

        all_ocr, vector_ids = [], []
        all_handwritten_text, all_printed_text = [], []
        all_handwritten_items, all_printed_items = [], []
        
        for i, img in enumerate(images):
            logger.info(f"Processing page {i+1}/{len(images)}...")
            pre = self.preprocessor.preprocess_image(img)
            ocr_res = self.ocr_engine.hybrid_extract(pre)
            all_ocr.append(ocr_res)
            
            if ocr_res.get("handwritten_text"):
                all_handwritten_text.append(ocr_res["handwritten_text"])
            if ocr_res.get("printed_text"):
                all_printed_text.append(ocr_res["printed_text"])
            if ocr_res.get("handwritten_items"):
                all_handwritten_items.extend(ocr_res["handwritten_items"])
            if ocr_res.get("printed_items"):
                all_printed_items.extend(ocr_res["printed_items"])
            
            page_meta = {
                "document_id": doc_id, 
                "page_number": i+1, 
                "source_file": os.path.basename(file_path), 
                "extraction_date": datetime.utcnow().isoformat()
            }
            vid = self.vector_db.add_document(doc_id, ocr_res.get("raw_text", ""), page_meta)
            if vid:
                vector_ids.append(vid)

        combined = "\n\n".join([r.get("raw_text", "") for r in all_ocr])
        combined_handwritten = "\n".join(all_handwritten_text)
        combined_printed = "\n".join(all_printed_text)
        avg_conf = sum([r.get("confidence", 0.0) for r in all_ocr]) / max(1, len(all_ocr))
        def clean_text(text):
            text = re.sub(r'[^A-Za-z0-9\s.,:/-]', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            return text.strip()

        combined_cleaned = clean_text(combined)
        handwritten_cleaned = clean_text(combined_handwritten)

        structured_data = self.llm_enricher.structure_medical_data(
            raw_text=combined_cleaned,
            handwritten_text=handwritten_cleaned
        )

        elapsed = (datetime.now() - start).total_seconds() * 1000

        metadata = {
            "extraction_method": "hybrid_ocr",
            "ocr_engines": ["PaddleOCR", "Tesseract"],
            "llm_model": self.llm_enricher.model_name,
            "confidence_scores": {"average": avg_conf},
            "processing_time_ms": elapsed,
            "vector_db_ids": vector_ids,
            "overall_confidence": avg_conf,
            "handwritten_regions": len(all_handwritten_items),
            "printed_regions": len(all_printed_items),
            "completeness": 1.0 if combined else 0.0,
            "handwritten_text": combined_handwritten,
            "printed_text": combined_printed,
            "handwritten_items": all_handwritten_items,
            "printed_items": all_printed_items
        }

        doc_info = {
            "document_id": doc_id,
            "document_type": "medical_record",
            "source_file": os.path.basename(file_path),
            "file_hash": file_hash,
            "page_count": len(images)
        }

        prover_json = ProverJSON.create(doc_info, structured_data, metadata)
        out_path = os.path.join(output_dir, f"{doc_id}_prover.json")

        # Save and verify
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(prover_json, f, indent=2, ensure_ascii=False)

        if os.path.exists(out_path):
            logger.info(f"✅ Saved Prover JSON: {out_path}")
        else:
            logger.error(f"❌ Could not find saved Prover JSON at: {out_path}")

        logger.info(f"✅ Processing complete in {elapsed/1000:.2f}s")

        return {
            "document_id": doc_id,
            "output_path": out_path,
            "document_header": prover_json.get("document_header"),
            "provenance": prover_json.get("provenance"),
            "quality_metrics": prover_json.get("quality_metrics"),
            "text_analysis": prover_json.get("text_analysis"),
            "handwritten_text": combined_handwritten,
            "printed_text": combined_printed,
            "prover_json": prover_json
        }
